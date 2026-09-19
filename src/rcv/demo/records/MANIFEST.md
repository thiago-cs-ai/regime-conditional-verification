# Provenance — the five published chain records

These records are the artifacts behind **Table T7, "The five chain records"** of
*Regime-Conditional Verification: Correctness Estimation for Adapting and Monitoring Safety
Classifiers* (Sandoval & Topcu, arXiv:2608.14089).

`rcv demo` replays these bundled JSON records from the published chain run. Each chain ends
with no alarm or a request for escalation.

Accepted repairs are probe updates that passed the held-out check. Total labels include audit
labels and one 300-label gate block per alarmed cycle. Final attack share is the percentage of the
final baseline traffic mixture coming from harm families incorporated after accepted repairs.

The last cycle's outcome is `no alarm` if the monitor did not fire, or `escalated` if no update
passed within the label budget.

| Chain | Last cycle outcome | Total labels | Accepted repairs | Final attack share |
|---|---|---|---|---|
| 42 | no alarm | 2,700 | 4 | 54.4% |
| 789 | no alarm | 1,800 | 3 | 42.5% |
| 1024 | escalated | 3,300 | 3 | 38.3% |
| 456 | escalated | 3,000 | 2 | 31.6% |
| 123 | escalated | 1,500 | 0 | 0.0% |

## SHA-256 checksums

| File | SHA-256 |
|---|---|
| `chain_42/CHAIN.json` | `2255a50849b25e602d48a18b55dd2056058f95e1500e852aa282119ef444af6f` |
| `chain_42/cycle_1.json` | `c48d044d4877675af82f1e4f782cf1825cf0edaa70ac0e729a4cf38c007835d3` |
| `chain_42/cycle_2.json` | `514e6e4df36c035444255e47df1b30b732a02c343e54e02f16177ee29ddbd461` |
| `chain_42/cycle_3.json` | `d32df81a69f5484dd7da9de84f78dd403c989e1b2738d03706d679ea75c7a1c6` |
| `chain_42/cycle_4.json` | `4516715e1291fe5dc435059c36001c62a4b76273930c3ae4afa0e9ceb727132e` |
| `chain_42/cycle_5.json` | `9bd5ac86f25f0c29cae52017955fa8d82380527fb9e103268052a3b05f567a57` |
| `chain_123/CHAIN.json` | `dd42b522518b7d7d9ebee7167aedb9b0f7f5ea192bf2437a7d9e3dc06deb2fd8` |
| `chain_123/cycle_1.json` | `1bed84ccfe3ad8aa0a1a7028edfce482752bf11bb566d5c22f1b40b85456a696` |
| `chain_456/CHAIN.json` | `b63bb780457272b2266470ef8c0909eb448d65f91685a33a6038e312f1f0ef76` |
| `chain_456/cycle_1.json` | `0bb0b0d889adc7da51ae85a6a0528422c2825cb232b382fa5fd9401359d3f792` |
| `chain_456/cycle_2.json` | `0f52ffdecb8ebf3090745cefde31885fa54df3892ae3e691b86c1171e355a945` |
| `chain_456/cycle_3.json` | `732a898af7a60542c4107e2fba86d16ee62a1a628a1851d480bc252c94204cb0` |
| `chain_789/CHAIN.json` | `e9a5199d3789b6bc57f1b760dc54c548cce538ecc4a08d05869c34feb3fd4037` |
| `chain_789/cycle_1.json` | `42519b36c93e278cbba0d2921cd2a8519c28b45bdb6c3799d2c08e67517dd16e` |
| `chain_789/cycle_2.json` | `8d466a4eb5e998d7c51377983e469a18a16e91646ade32f0f76e32b699907b0a` |
| `chain_789/cycle_3.json` | `9970a1644bcb620ff791f7c464cfd064721e22b64b40d1e6ae776b69e6992e19` |
| `chain_789/cycle_4.json` | `47080b0ee52023d6154e7e477a727bf54b54400eceb7eba0d087ff748e7f1ee0` |
| `chain_1024/CHAIN.json` | `1e75ed8cadd6efa5ff66e195600e348420b1ffe13a5f9709b8797f392ec2acbb` |
| `chain_1024/cycle_1.json` | `e9de9a1e5962ba2d5c005421bda82ba0d6fb7393e24452328c8457519ff51e4a` |
| `chain_1024/cycle_2.json` | `8980744e52e543519c1ce8aee703c8ba4bc2d5c536216883b5716e5e49bdfae5` |
| `chain_1024/cycle_3.json` | `afc8e20635ecb55c688f1157bfef706303450d819ed61b4994247bd2d9fc4846` |
| `chain_1024/cycle_4.json` | `1a8a727ba8966406c0c93b09a0c475ca4e00ca80d91e466608d586a042264577` |
