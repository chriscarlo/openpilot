package main

import (
	"testing"

	"capnproto.org/go/capnp/v3"
)

func makeTestWay(t *testing.T, forward, backward CompactWindingSummary) Way {
	t.Helper()
	msg, seg, err := capnp.NewMessage(capnp.SingleSegment(nil))
	if err != nil {
		t.Fatalf("NewMessage failed: %v", err)
	}
	way, err := NewRootWay(seg)
	if err != nil {
		t.Fatalf("NewRootWay failed: %v", err)
	}
	way.SetWindingForwardLevel(forward.Level)
	way.SetWindingForwardScore(forward.Score)
	way.SetWindingForwardConfidence(forward.Confidence)
	way.SetWindingBackwardLevel(backward.Level)
	way.SetWindingBackwardScore(backward.Score)
	way.SetWindingBackwardConfidence(backward.Confidence)
	if _, err := ReadRootWay(msg); err != nil {
		t.Fatalf("ReadRootWay failed: %v", err)
	}
	return way
}

func TestDirectionalWindingSummaryUsesDirectionSpecificFields(t *testing.T) {
	way := makeTestWay(t,
		CompactWindingSummary{Level: 4, Score: 220, Confidence: 200},
		CompactWindingSummary{Level: 1, Score: 70, Confidence: 160},
	)

	forward := DirectionalWindingSummary(way, true)
	backward := DirectionalWindingSummary(way, false)

	if forward.Level != 4 || forward.Score != 220 || forward.Confidence != 200 {
		t.Fatalf("unexpected forward summary: %+v", forward)
	}
	if backward.Level != 1 || backward.Score != 70 || backward.Confidence != 160 {
		t.Fatalf("unexpected backward summary: %+v", backward)
	}
}

func TestAggregateRouteWindingSummaryPreservesCurrentAndBoostsCluster(t *testing.T) {
	current := CompactWindingSummary{Level: 2, Score: 110, Confidence: 180}
	future := []CompactWindingSummary{
		{Level: 3, Score: 150, Confidence: 200},
		{Level: 4, Score: 210, Confidence: 210},
	}

	summary := AggregateRouteWindingSummary(current, future)

	if !summary.Valid {
		t.Fatal("expected valid route summary")
	}
	if summary.CurrentLevel != 2 || summary.CurrentScore != 110 || summary.CurrentConfidence != 180 {
		t.Fatalf("unexpected current summary fields: %+v", summary)
	}
	if summary.Level != 4 {
		t.Fatalf("expected max horizon level 4, got %d", summary.Level)
	}
	if summary.Score <= 210 {
		t.Fatalf("expected combined route score to exceed strongest single-way score, got %d", summary.Score)
	}
	if summary.Confidence <= 210 {
		t.Fatalf("expected combined confidence to exceed strongest single-way confidence, got %d", summary.Confidence)
	}
	if summary.WayCount != 3 {
		t.Fatalf("expected way count 3, got %d", summary.WayCount)
	}
}

func TestAggregateRouteWindingSummaryStaysInvalidWhenTilesLackMetadata(t *testing.T) {
	summary := AggregateRouteWindingSummary(CompactWindingSummary{}, []CompactWindingSummary{{}, {}})

	if summary.Valid {
		t.Fatal("expected invalid summary when all directional metadata is absent")
	}
	if summary.Level != 0 || summary.Score != 0 || summary.Confidence != 0 {
		t.Fatalf("expected zero horizon summary, got %+v", summary)
	}
	if summary.WayCount != 3 {
		t.Fatalf("expected way count to reflect traversed chain, got %d", summary.WayCount)
	}
}
