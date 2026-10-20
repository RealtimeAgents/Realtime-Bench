#!/bin/bash
set -e
service_pids=()
trap 'rm -f /tmp/sandbox-ready' EXIT

echo "[entrypoint] Starting sandbox environment..."

# Hold the launcher-provided lane lock across the complete instance lifetime.
exec 9>/opt/sandbox-network.lock
flock -n 9 || { echo "Another sandbox is already using this lane." >&2; exit 1; }
export HOME=/home/user
export DISPLAY="${DISPLAY:-:99}"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_MENU_PREFIX=xfce-
export DESKTOP_SESSION=xfce
export XDG_CURRENT_DESKTOP=XFCE
# CDP provides accessibility data; disabling AT-SPI avoids another unnecessary
# per-session socket service in concurrent desktop lanes.
export NO_AT_BRIDGE=1
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
cd "$HOME"
dbus-daemon --session --address="$DBUS_SESSION_BUS_ADDRESS" --nofork --nopidfile &
service_pids+=("$!")
sleep 1

# Start Xvfb (virtual framebuffer). DISPLAY is lane-local because X11 also uses
# an abstract socket that is not isolated by a private /tmp bind.
echo "[entrypoint] Starting Xvfb on ${DISPLAY}..."
Xvfb "$DISPLAY" -screen 0 1920x1080x24 -ac -nolisten tcp &
service_pids+=("$!")
sleep 2

# Start the XFCE desktop components directly. xfce4-session creates PID-derived
# ICE abstract sockets, which can collide when separate PID namespaces share the
# host network namespace. These components provide the same benchmark desktop
# without the session-manager socket.
echo "[entrypoint] Starting XFCE desktop..."
xrdb -merge /etc/xdg/xfce4/Xft.xrdb 2>/dev/null || true
xfwm4 --compositor=off &
service_pids+=("$!")
xfsettingsd --disable-wm-check &
service_pids+=("$!")
xfce4-panel &
service_pids+=("$!")
xfdesktop --disable-wm-check &
service_pids+=("$!")
sleep 3

# Start VNC server
echo "[entrypoint] Starting x11vnc..."
# The packaged x11vnc enables a separate IPv6 wildcard listener unless -no6 is explicit.
x11vnc -display "$DISPLAY" -forever -nopw -no6 -listen 127.0.0.1 -rfbport "${VNC_PORT:-5900}" -shared &
service_pids+=("$!")
sleep 1

# Start noVNC (web-based VNC viewer)
echo "[entrypoint] Starting noVNC..."
websockify --web /usr/share/novnc "127.0.0.1:${NOVNC_PORT:-5910}" "127.0.0.1:${VNC_PORT:-5900}" &
service_pids+=("$!")
sleep 1

# Start guest agent server
echo "[entrypoint] Starting guest agent server..."
cd /opt/guest_agent && python3 server.py &
service_pids+=("$!")
sleep 1

# Do not report readiness for a failed child or a service using a conflicting port.
for pid in "${service_pids[@]}"; do kill -0 "$pid"; done
xdpyinfo -display "$DISPLAY" >/dev/null
curl --retry 10 --retry-connrefused --retry-delay 1 --fail --silent \
  "http://127.0.0.1:${GUEST_AGENT_PORT:-5000}/health" >/dev/null
for pid in "${service_pids[@]}"; do kill -0 "$pid"; done
touch /tmp/sandbox-ready
echo "[entrypoint] All services started."
echo "[entrypoint] noVNC: http://localhost:${NOVNC_PORT:-5910}/vnc.html"
echo "[entrypoint] Guest Agent: http://localhost:${GUEST_AGENT_PORT:-5000}"

# Keep container running
wait -n
