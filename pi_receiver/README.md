# PMU UART Receiver

Build the receiver on the Raspberry Pi with `make`, then capture the MCXN947 debug UART at 115200 baud:

```bash
./pmu_uart_receiver -d /dev/ttyACM0 -o mcxn947_raw.csv
```

It validates the `PMU3` packet magic, protocol version, fixed two-channel/128-sample payload, and CRC-16/CCITT-FALSE. It reports CRC failures, sequence gaps, and non-monotonic timestamps while storing `voltage_raw` and `current_raw` ADC codes in CSV.