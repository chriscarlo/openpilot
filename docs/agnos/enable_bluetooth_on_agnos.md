Goal
- Enable Bluetooth on AGNOS for comma 3/3X by adding BlueZ userspace and QCA firmware via a minimal agnos-builder change. Kernel/DT already have WCN3990 support built-in on device.

Patch (unified diff)
- Save the block below as bluetooth_on_agnos.patch and apply it in a clone of agnos-builder.

```
diff --git a/Dockerfile.agnos b/Dockerfile.agnos
index 0000000..0000000 100644
--- a/Dockerfile.agnos
+++ b/Dockerfile.agnos
@@ -116,6 +116,10 @@ FROM agnos-base
 RUN cd /tmp && \
     apt-get update && \
     apt-get install -yq --no-install-recommends \
     python3 \
     python3-dev \
     gir1.2-qmi-1.0 \
     libglib2.0-dev \
     libqmi-glib5 \
     libc6 \
     libglib2.0-0t64 \
     libgudev-1.0-0 \
     libmm-glib0 \
     libpolkit-gobject-1-0 \
     libsystemd0 \
     polkitd \
     mobile-broadband-provider-info && \
     apt-get -o Dpkg::Options::="--force-overwrite" install -yq ./libqmi.deb && \
     apt-get -o Dpkg::Options::="--force-overwrite" install -yq ./modemmanager.deb && \
     apt-get -o Dpkg::Options::="--force-overwrite" install -yq ./lpac.deb
@@ -150,6 +154,20 @@ ARG XDG_DATA_HOME="/usr/local"
     MAKEFLAGS="-j$(nproc)" UV_NO_CACHE=1 UV_PROJECT_ENVIRONMENT=$XDG_DATA_HOME/venv uv sync --frozen --inexact --compile-bytecode
 
 # Install nice to haves
 COPY ./userspace/install_extras.sh /tmp/agnos/
 RUN /tmp/agnos/install_extras.sh

+# Bluetooth support (userspace + firmware)
+# - bluez: provides bluetoothd, btmgmt, bluetoothctl, udev rules, systemd unit
+# - rfkill: troubleshooting tool; harmless overhead
+# - linux-firmware: includes QCA WCN3990 BT firmware under /lib/firmware/qca
+RUN apt-get update && \
+    apt-get install -yq --no-install-recommends \
+      bluez \
+      rfkill \
+      linux-firmware && \
+    rm -rf /var/lib/apt/lists/*

 COPY --from=agnos-compiler-qtwayland5 /tmp/qtwayland5.deb /tmp/qtwayland5.deb
 RUN cd /tmp && apt-get -o Dpkg::Options::="--force-overwrite" install -yq --allow-downgrades ./qtwayland5.deb
 
 # Patched libeglSubDriverWayland with fixed nullptr deref in CommitBuffer
diff --git a/userspace/services.sh b/userspace/services.sh
index 0000000..0000000 100755
--- a/userspace/services.sh
+++ b/userspace/services.sh
@@ -1,6 +1,10 @@
 #!/bin/bash -e
 
 # Enable DSP support services
 systemctl enable adsp
 systemctl enable cdsp
 systemctl enable adsprpcd
 systemctl enable cdsprpcd
+
+# Enable Bluetooth service (provided by bluez)
+# Kernel/DT contain WCN3990 support; userspace + firmware come from packages above
+systemctl enable bluetooth
 
 # Enable our services
 systemctl enable fs_setup.service
 systemctl enable serial-hostname.service
```

Build steps
- Host prerequisites: Docker with buildx (>=0.15), network access.
- Clone and build system only (kernel already has BT built-in on the device you checked):
- Commands:
  - git clone https://github.com/commaai/agnos-builder.git
  - cd agnos-builder
  - git submodule update --init agnos-kernel-sdm845
  - patch -p1 < ../bluetooth_on_agnos.patch
  - ./tools/extract_tools.sh
  - ./build_system.sh

Flash (does not wipe /data or /persist)
- Put device in EDL/QDL mode (see https://flash.comma.ai for details).
- From the agnos-builder repo on your host:
  - ./flash_system.sh
- This overwrites only the active system_a or system_b partition. Kernel and bootloaders remain unchanged.

Post-flash verification (on device)
- Check firmware presence:
  - ls /lib/firmware/qca | head
- Confirm service is enabled and running:
  - systemctl status bluetooth --no-pager
- Kernel already exposes WCN3990 (from your earlier check). Bring up controller:
  - rfkill list; rfkill unblock all
  - btmgmt --index 0 info || hciconfig -a
  - btmgmt --index 0 power on
  - bluetoothctl show
- dmesg diagnostics:
  - dmesg | rg -i 'bluetooth|qca|wcn|btfm'

Rollback
- Reflash the previous release with scripts/download-from-manifest.py (agnos-builder) or use https://flash.comma.ai.

Notes
- If linux-firmware in your environment doesn’t include the exact WCN3990 BT blobs, drop them into agnos-builder/userspace/firmware/qca/ and add a COPY layer similar to the existing firmware COPY. The Dockerfile already copies userspace/firmware/* into /lib/firmware.
- If you prefer testing without rebuilding, you can temporarily place bluetoothd and firmware under /data and run bluetoothd manually, but persistence and systemd integration are best via the system image approach above.

