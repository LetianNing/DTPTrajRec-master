# DTPTrajRec

## Environment Setup

- **Python**: 3.8.10
- **PyTorch**: 1.10.0
- **NumPy**: 1.24.4
- **nni**: 3.0
- **rtree**: 1.3.0

## Dataset

We provide sample data under the `./data/` directory. 

- **Road Network Structure**: `./data/extra_data/road_network`
- **Weighted Directed Graph Structure**: `./data/extra_data/TLG`

Full datasets obtained in the baseline [MM-STGED](https://dl.acm.org/doi/10.1109/TKDE.2024.3396158).

The roles of other files are as follows:

- **Chengdu_SE.txt**: Embedding representation of road segments.
- **new2raw_rid.json**: Mapping from new ID to raw ID.
- **raw2new_rid.json**: Mapping from raw ID to new ID.
- **raw_rn_dict.json**: Attributes of road segments corresponding to raw ID, including coordinates, length, level, etc.
- **rn_dict.json**: Attributes of road segments corresponding to new ID, including coordinates, length, level, etc.
- **uid2index.json**: Mapping from raw user ID to new user ID.

Each line the the trajectory consists(a example):
- raw trajectory $T$(observation window is $\epsilon$): p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26
- $\delta$-samping trajectory $T^\delta$: p1 p6 p11 p16 p21 p26
- $T$:p1 * * * * p6 * * * * p11 * * * * p16 * * * * p21 * * * * p26
- $T_f$: p1 p1 p1 p1 p1 p6 p6 p6 p6 p6 p11 p11 p11 p11 p11 p16 p16 p16 p16 p16 p21 p21 p21 p21 p21 p26
- $T_d$: p1 p6 p6 p6 p6 p6 p11 p11 p11 p11 p11 p16 p16 p16 p16 p16 p21 p21 p21 p21 p21 p26 p26 p26 p26 p26

## Running

To run the code, use the following command:
> python multi_main.py