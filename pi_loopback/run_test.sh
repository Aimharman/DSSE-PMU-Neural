#!/usr/bin/env bash
# run_test.sh - run transmitter + receiver together and capture logs for diagnosis.
#
# Usage: sudo ./run_test.sh [out_gpio] [in_gpio] [sine_hz] [duration_s]
set -euo pipefail

OUT_GPIO="${1:-18}"
IN_GPIO="${2:-23}"
SINE_HZ="${3:-50}"
DURATION="${4:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TX_LOG="tx.log"
RX_CSV="capture.csv"

echo "Starting transmitter on gpio${OUT_GPIO} (sine ${SINE_HZ}Hz, ${DURATION}s)..."
./transmitter -g "$OUT_GPIO" -f "$SINE_HZ" -c 20000 -a 1.0 -r 1000 -d "$DURATION" > "$TX_LOG" 2>&1 &
TX_PID=$!

# give the transmitter a moment to initialise pigpio and start PWM
sleep 0.5

if ! kill -0 "$TX_PID" 2>/dev/null; then
    echo "Transmitter exited immediately - check $TX_LOG:"
    cat "$TX_LOG"
    exit 1
fi

echo "Starting receiver on gpio${IN_GPIO} -> $RX_CSV ..."
./receiver -g "$IN_GPIO" -a 1.0 -r 1000 -d "$DURATION" -o "$RX_CSV"

wait "$TX_PID" || true

echo "--- transmitter log ---"
cat "$TX_LOG"
echo "--- done, output: $RX_CSV ---"
