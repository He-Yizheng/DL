# Revisiting Differential-Linear Cryptanalysis via Set Propagation

This repository contains the anonymized source code accompanying our paper *Revisiting Differential-Linear Cryptanalysis via Set Propagation*.

## Code Availability

The WARP directory provides a complete implementation and serves as the reference implementation of the proposed DL-BSP and DL-PSP frameworks. For the other evaluated ciphers, the repository provides the key cipher-specific code used for set propagation, correlation evaluation, and experimental verification. These directories contain the core implementation components but do not constitute complete pipelines.

| Cipher | Code availability |
| --- | --- |
| WARP | Complete  implementation |
| ASCON | Key implementation components |
| SKINNY-128 | Key implementation components |
| LBlock | Key implementation components |
| TWINE | Key implementation components |

The provided materials are intended to clarify the implementation details and facilitate verification of the main results reported in the paper.
