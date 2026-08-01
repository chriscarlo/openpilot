#include <sys/xattr.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/params.h"
#include "system/loggerd/encoder/encoder.h"
#include "system/loggerd/loggerd.h"
#include "system/loggerd/video_writer.h"

ExitHandler do_exit;

// How long after a bookmark/preserve trigger a newly-rotated-into segment still
// counts as part of that incident. The onroad flag button advertises a +6.0 s
// post window (kFlagRetroPostS, selfdrive/ui/sunnypilot/qt/onroad/buttons.h);
// this is that window plus margin for UI->msgq->loggerd latency. A tap in the
// last few seconds of a segment records its post window into segment N+1, and
// N+1 does not exist yet at tap time -- so it cannot be xattr'd by
// handle_preserve_segment(). Instead we arm a deadline and xattr the next
// segment when we actually rotate into it. (deleter.py only ever expands a
// preserved segment BACKWARDS: range(max(0, seg_num - 2), seg_num + 1), so
// without this the post window sits in an unprotected, reapable segment.)
//
// Kept tight on purpose (6.0 s advertised + 2.0 s of margin, not 2x): every
// extra xattr'd directory consumes one of deleter.py's PRESERVE_COUNT slots.
constexpr double PRESERVE_POST_WINDOW_MS = 8000.0;

struct LoggerdState {
  LoggerState logger;
  std::atomic<double> last_camera_seen_tms{0.0};
  std::atomic<int> ready_to_rotate{0};  // count of encoders ready to rotate
  int max_waiting = 0;
  double last_rotate_tms = 0.;      // last rotate time in ms
  double preserve_until_tms = 0.;   // xattr any segment we rotate into before this time
  int preserved_segment = -1;       // last segment already xattr'd, dedup only
};

// Mark the CURRENT segment as preserved. Idempotent per segment.
void preserve_current_segment(LoggerdState *s) {
  if (s->logger.segment() == s->preserved_segment) return;

  LOGW("preserving %s", s->logger.segmentPath().c_str());

#ifdef __APPLE__
  int ret = setxattr(s->logger.segmentPath().c_str(), PRESERVE_ATTR_NAME, &PRESERVE_ATTR_VALUE, 1, 0, 0);
#else
  int ret = setxattr(s->logger.segmentPath().c_str(), PRESERVE_ATTR_NAME, &PRESERVE_ATTR_VALUE, 1, 0);
#endif
  if (ret) {
    LOGE("setxattr %s failed for %s: %s", PRESERVE_ATTR_NAME, s->logger.segmentPath().c_str(), strerror(errno));
  }

  // mark route for uploading
  Params params;
  std::string routes = params.get("AthenadRecentlyViewedRoutes");
  params.put("AthenadRecentlyViewedRoutes", routes + "," + s->logger.routeName());

  s->preserved_segment = s->logger.segment();
}

void logger_rotate(LoggerdState *s) {
  bool ret =s->logger.next();
  assert(ret);
  s->ready_to_rotate = 0;
  s->last_rotate_tms = millis_since_boot();
  LOGW((s->logger.segment() == 0) ? "logging to %s" : "rotated to %s", s->logger.segmentPath().c_str());

  // Forward half of the preserve window: we just rotated into the segment that
  // holds the tail of a recent bookmark's post window, so protect it too.
  if (s->last_rotate_tms < s->preserve_until_tms) {
    preserve_current_segment(s);
  }
}

void rotate_if_needed(LoggerdState *s) {
  // all encoders ready, trigger rotation
  bool all_ready = s->ready_to_rotate == s->max_waiting;

  // fallback logic to prevent extremely long segments in the case of camera, encoder, etc. malfunctions
  bool timed_out = false;
  double tms = millis_since_boot();
  double seg_length_secs = (tms - s->last_rotate_tms) / 1000.;
  if ((seg_length_secs > SEGMENT_LENGTH) && !LOGGERD_TEST) {
    // TODO: might be nice to put these reasons in the sentinel
    if ((tms - s->last_camera_seen_tms) > NO_CAMERA_PATIENCE) {
      timed_out = true;
      LOGE("no camera packets seen. auto rotating");
    } else if (seg_length_secs > SEGMENT_LENGTH*1.2) {
      timed_out = true;
      LOGE("segment too long. auto rotating");
    }
  }

  if (all_ready || timed_out) {
    logger_rotate(s);
  }
}

struct RemoteEncoder {
  std::unique_ptr<VideoWriter> writer;
  int encoderd_segment_offset;
  int current_segment = -1;
  std::vector<Message *> q;
  int dropped_frames = 0;
  bool recording = false;
  bool marked_ready_to_rotate = false;
  bool seen_first_packet = false;
  bool audio_initialized = false;
};

size_t write_encode_data(LoggerdState *s, cereal::Event::Reader event, RemoteEncoder &re, const EncoderInfo &encoder_info) {
  auto edata = (event.*(encoder_info.get_encode_data_func))();
  auto idx = edata.getIdx();
  auto flags = idx.getFlags();

  // if we aren't recording yet, try to start, since we are in the correct segment
  if (!re.recording) {
    if (flags & V4L2_BUF_FLAG_KEYFRAME) {
      // only create on iframe
      if (re.dropped_frames) {
        // this should only happen for the first segment, maybe
        LOGW("%s: dropped %d non iframe packets before init", encoder_info.publish_name, re.dropped_frames);
        re.dropped_frames = 0;
      }
      if (encoder_info.record) {
        // write the header
        auto header = edata.getHeader();
        re.writer->write((uint8_t *)header.begin(), header.size(), idx.getTimestampEof() / 1000, true, false);
      }
      re.recording = true;
    } else {
      // this is a sad case when we aren't recording, but don't have an iframe
      // nothing we can do but drop the frame
      ++re.dropped_frames;
      return 0;
    }
  }

  // we have to be recording if we are here
  assert(re.recording);

  // if we are actually writing the video file, do so
  if (re.writer) {
    auto data = edata.getData();
    re.writer->write((uint8_t *)data.begin(), data.size(), idx.getTimestampEof() / 1000, false, flags & V4L2_BUF_FLAG_KEYFRAME);
  }

  // put it in log stream as the idx packet
  MessageBuilder bmsg;
  auto evt = bmsg.initEvent(event.getValid());
  evt.setLogMonoTime(event.getLogMonoTime());
  (evt.*(encoder_info.set_encode_idx_func))(idx);
  auto new_msg = bmsg.toBytes();
  s->logger.write((uint8_t *)new_msg.begin(), new_msg.size(), true);  // always in qlog?
  return new_msg.size();
}

int handle_encoder_msg(LoggerdState *s, Message *msg, std::string &name, struct RemoteEncoder &re, const EncoderInfo &encoder_info) {
  int bytes_count = 0;

  // extract the message
  capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize() / sizeof(capnp::word)));
  auto event = cmsg.getRoot<cereal::Event>();
  auto edata = (event.*(encoder_info.get_encode_data_func))();
  auto idx = edata.getIdx();

  // encoderd can have started long before loggerd
  if (!re.seen_first_packet) {
    re.seen_first_packet = true;
    re.encoderd_segment_offset = idx.getSegmentNum();
    LOGD("%s: has encoderd offset %d", name.c_str(), re.encoderd_segment_offset);
  }
  int offset_segment_num = idx.getSegmentNum() - re.encoderd_segment_offset;

  if (offset_segment_num == s->logger.segment()) {
    // loggerd is now on the segment that matches this packet

    // if this is a new segment, we close any possible old segments, move to the new, and process any queued packets
    if (re.current_segment != s->logger.segment()) {
      // if we aren't actually recording, don't create the writer
      if (encoder_info.record) {
        assert(encoder_info.filename != NULL);
        re.writer.reset(new VideoWriter(s->logger.segmentPath().c_str(),
                                        encoder_info.filename, idx.getType() != cereal::EncodeIndex::Type::FULL_H_E_V_C,
                                        edata.getWidth(), edata.getHeight(), encoder_info.fps, idx.getType()));
        re.recording = false;
        re.audio_initialized = false;
      }
      re.current_segment = s->logger.segment();
      re.marked_ready_to_rotate = false;
    }
    if (re.audio_initialized || !encoder_info.include_audio) {
      // we are in this segment now, process any queued messages before this one
      if (!re.q.empty()) {
        for (auto qmsg : re.q) {
          capnp::FlatArrayMessageReader reader({(capnp::word *)qmsg->getData(), qmsg->getSize() / sizeof(capnp::word)});
          bytes_count += write_encode_data(s, reader.getRoot<cereal::Event>(), re, encoder_info);
          delete qmsg;
        }
        re.q.clear();
      }
      bytes_count += write_encode_data(s, event, re, encoder_info);
      delete msg;
    } else if (re.q.size() > MAIN_FPS*10) {
      LOGE_100("%s: dropping frame waiting for audio initialization, queue is too large", name.c_str());
      delete msg;
    } else {
      re.q.push_back(msg); // queue up all the new segment messages, they go in after audio is initialized
    }
  } else if (offset_segment_num > s->logger.segment()) {
    // encoderd packet has a newer segment, this means encoderd has rolled over
    if (!re.marked_ready_to_rotate) {
      re.marked_ready_to_rotate = true;
      ++s->ready_to_rotate;
      LOGD("rotate %d -> %d ready %d/%d for %s",
        s->logger.segment(), offset_segment_num,
        s->ready_to_rotate.load(), s->max_waiting, name.c_str());
    }

    // TODO: define this behavior, but for now don't leak
    if (re.q.size() > MAIN_FPS*10) {
      LOGE_100("%s: dropping frame, queue is too large", name.c_str());
      delete msg;
    } else {
      // queue up all the new segment messages, they go in after the rotate
      re.q.push_back(msg);
    }
  } else {
    LOGE("%s: encoderd packet has a older segment!!! idx.getSegmentNum():%d s->logger.segment():%d re.encoderd_segment_offset:%d",
      name.c_str(), idx.getSegmentNum(), s->logger.segment(), re.encoderd_segment_offset);
    // free the message, it's useless. this should never happen
    // actually, this can happen if you restart encoderd
    re.encoderd_segment_offset = -s->logger.segment();
    delete msg;
  }

  return bytes_count;
}

void handle_preserve_segment(LoggerdState *s) {
  // Arm the forward window BEFORE the dedup return: a second tap late in a
  // segment we already preserved must still extend protection into N+1.
  const double deadline = millis_since_boot() + PRESERVE_POST_WINDOW_MS;
  if (deadline > s->preserve_until_tms) {
    s->preserve_until_tms = deadline;
  }

  preserve_current_segment(s);
}

void loggerd_thread() {
  // setup messaging
  typedef struct ServiceState {
    std::string name;
    int counter, freq;
    bool encoder, preserve_segment, record_audio;
  } ServiceState;
  std::unordered_map<SubSocket*, ServiceState> service_state;
  std::unordered_map<SubSocket*, struct RemoteEncoder> remote_encoders;

  std::unique_ptr<Context> ctx(Context::create());
  std::unique_ptr<Poller> poller(Poller::create());

  // subscribe to all socks
  for (const auto& [_, it] : services) {
    const bool encoder = util::ends_with(it.name, "EncodeData");
    const bool livestream_encoder = util::starts_with(it.name, "livestream");
    const bool record_audio = (it.name == "rawAudioData") && Params().getBool("RecordAudio");
    if (it.should_log || (encoder && !livestream_encoder) || record_audio) {
      LOGD("logging %s", it.name.c_str());

      SubSocket * sock = SubSocket::create(ctx.get(), it.name);
      assert(sock != NULL);
      poller->registerSocket(sock);
      service_state[sock] = {
        .name = it.name,
        .counter = 0,
        .freq = it.decimation,
        .encoder = encoder,
        // "bookmarkButton" is the raw tap; "userBookmark" is feedbackd's echo of it.
        // Preserving on the raw tap makes deletion-protection independent of feedbackd,
        // which is registered only_onroad (system/manager/process_config.py) -- if it is
        // down or crashes, the tap would otherwise be recorded into a segment that
        // deleter is still free to reap.
        .preserve_segment = (it.name == "userBookmark") || (it.name == "bookmarkButton") || (it.name == "audioFeedback"),
        .record_audio = record_audio,
      };
    }
  }

  LoggerdState s;
  // init logger
  logger_rotate(&s);
  Params().put("CurrentRoute", s.logger.routeName());

  std::map<std::string, EncoderInfo> encoder_infos_dict;
  std::vector<RemoteEncoder*> encoders_with_audio;
  for (const auto &cam : cameras_logged) {
    for (const auto &encoder_info : cam.encoder_infos) {
      encoder_infos_dict[encoder_info.publish_name] = encoder_info;
      s.max_waiting++;
    }
  }

  for (auto &[sock, service] : service_state) {
    auto it = encoder_infos_dict.find(service.name);
    if (it != encoder_infos_dict.end() && it->second.include_audio) {
      encoders_with_audio.push_back(&remote_encoders[sock]);
    }
  }

  uint64_t msg_count = 0, bytes_count = 0;
  double start_ts = millis_since_boot();
  while (!do_exit) {
    // poll for new messages on all sockets
    for (auto sock : poller->poll(1000)) {
      if (do_exit) break;

      ServiceState &service = service_state[sock];
      if (service.preserve_segment) {
        handle_preserve_segment(&s);
      }

      // drain socket
      int count = 0;
      Message *msg = nullptr;
      while (!do_exit && (msg = sock->receive(true))) {
        const bool in_qlog = service.freq != -1 && (service.counter++ % service.freq == 0);

        if (service.record_audio) {
          capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize() / sizeof(capnp::word)));
          auto event = cmsg.getRoot<cereal::Event>();
          auto audio_data = event.getRawAudioData().getData();
          auto sample_rate = event.getRawAudioData().getSampleRate();
          for (auto* encoder : encoders_with_audio) {
            if (encoder && encoder->writer) {
              encoder->writer->write_audio((uint8_t*)audio_data.begin(), audio_data.size(), event.getLogMonoTime() / 1000, sample_rate);
              encoder->audio_initialized = true;
            }
          }
        }

        if (service.encoder) {
          s.last_camera_seen_tms = millis_since_boot();
          bytes_count += handle_encoder_msg(&s, msg, service.name, remote_encoders[sock], encoder_infos_dict[service.name]);
        } else {
          s.logger.write((uint8_t *)msg->getData(), msg->getSize(), in_qlog);
          bytes_count += msg->getSize();
          delete msg;
        }

        rotate_if_needed(&s);

        if ((++msg_count % 10000) == 0) {
          double seconds = (millis_since_boot() - start_ts) / 1000.0;
          LOGD("%" PRIu64 " messages, %.2f msg/sec, %.2f KB/sec", msg_count, msg_count / seconds, bytes_count * 0.001 / seconds);
        }

        count++;
        if (count >= 200) {
          LOGD("large volume of '%s' messages", service.name.c_str());
          break;
        }
      }
    }
  }

  LOGW("closing logger");
  s.logger.setExitSignal(do_exit.signal);

  if (do_exit.power_failure) {
    LOGE("power failure");
    sync();
    LOGE("sync done");
  }

  // messaging cleanup
  for (auto &[sock, service] : service_state) delete sock;
}

int main(int argc, char** argv) {
  if (!Hardware::PC()) {
    int ret;
    ret = util::set_core_affinity({0, 1, 2, 3});
    assert(ret == 0);
    // TODO: why does this impact camerad timings?
    //ret = util::set_realtime_priority(1);
    //assert(ret == 0);
  }

  loggerd_thread();

  return 0;
}
