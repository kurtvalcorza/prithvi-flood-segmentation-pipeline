"""Labelled-chip dataset contract for adapting the flood model: the pinned Sen1Floods11 sample, role assignment
from the official splits, BYOD loaders and sample export.

The default dataset is **real**: 44 hand-labelled 512 × 512 Sentinel-2 chips of Sen1Floods11 (Bonafilia et al.,
2020) — 24 from the official training split, 8 from the validation split and 12 from the test split, drawn
round-robin over the ten flood-event regions with a fixed seed on 2026-09-19 from the chips whose hand label is at
least 60 % valid (not cloud / no-data) and at least 3 % water, so every chip can be scored — pinned here by object
path, byte size and SHA-256 of both the 13-band `S2Hand` GeoTIFF and its `LabelHand` mask. Every object is fetched from the
public Sen1Floods11 bucket at run time and refused on any byte-size or SHA-256 mismatch; the repository
redistributes none of the chips. The roles follow the upstream splits. The 12 test-split chips were scored in every
earlier run and read during the 2026-09-19 build sweep, so they are the "exposed development test"; they are held
out only at chip level (all their events are in the upstream training split, several touch training chips).

The FRESH TEST (docs/prithvi-flood-preregistration.md) is 23 further official test-split chips that this notebook
never scored, pinned in `FRESH_TEST_RECORDS`: the 78 test chips outside the sample, filtered by the same label rule
(>= 60 % valid, >= 3 % water) to 25, minus the two upstream example chips the notebook displays. They come from the
same 8 events as the training data and most lie within one chip width of a training or validation chip, so they test
within-event generalisation, not new events.

A record is ``{id, image, label}``: a (6, 512, 512) reflectance array (or a GeoTIFF path) and a (512, 512) mask
with 0 = no water, 1 = water, -1 = no data / cloud (or a GeoTIFF path).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import (
    IMAGE_SIZE,
    MIN_RECORDS,
    MODEL_ID,
    S2_L1C_BAND_INDICES,
    atomic_write_bytes,
    check_record,
    chip_digest,
    read_chip,
    read_mask,
    validate_dataset,
)

CORPUS_NAME = "Sen1Floods11 hand-labelled Sentinel-2 chips (v1.1)"
CORPUS_RELEASE = "Sen1Floods11 v1.1 public bucket, 44 chips selected 2026-09-19 from the official hand-labelled splits"
CORPUS_BASE_URL = "https://storage.googleapis.com/sen1floods11/v1.1/"
CORPUS_LICENSE = "CC BY 4.0 (Cloud to Street; Bonafilia et al., CVPRW 2020)"
CORPUS_BYTES = 103_757_095
DEFAULT_CACHE_DIR = Path("weights") / "sen1floods11"
ROLES = ("train", "validation", "test")
# (chip name, role from the official split, S2Hand bytes, S2Hand sha256, LabelHand bytes, LabelHand sha256)
SAMPLE_RECORDS: tuple[tuple[str, str, int, str, int, str], ...] = (
    (
        "Ghana_141910",
        "train",
        2287617,
        "9d47e96a72447591200939350171395aa0c0a7c4d25e135d354c90e8889f1d6b",
        3562,
        "e6dd6c0eb8f2bc5e5e64b12e99f7398d27a0e56a4670f8be5382055504c4602b",
    ),
    (
        "Ghana_362274",
        "train",
        2235356,
        "76bea781cdd9ab8dc3992489709d0af14ffb47f2a5ff933b2e678f29db4d7e34",
        4129,
        "2030e61f58a19ed8a58359281959172a044ee4f45e589ff13555a55d9cb90ff9",
    ),
    (
        "Ghana_887131",
        "train",
        2212964,
        "fc41fa6e4187aef78ae0a1469c4ca483c21e09a304cfdd29c24e18a3d59f5fc1",
        5064,
        "f445c0716feb0ca1790b0c9ec3f20b067d36c99324d4632b4332ec2372797138",
    ),
    (
        "India_591549",
        "train",
        2403439,
        "b515458853602595b09331252532e40b38a2798733f8f8f87ceb87d3129dd060",
        8973,
        "ff69850ea4d2557bbbc6dd62bb7e8752725747f156055e431f2cea228209add2",
    ),
    (
        "India_91379",
        "train",
        2307403,
        "e06d91b2a236d8fe30a46ab96708119c9c437d1b40a54b2a766fbcd63efe271d",
        12942,
        "cac872eb9b59c21754d465033cb79cab91b66f833ca8a832d1150bb9acae088b",
    ),
    (
        "India_943439",
        "train",
        2255144,
        "7a3bd870a2e7723d9cbc348a565d974d815c6b999e9de984533e105dcc3bdab1",
        12360,
        "444d84e29ad06cbba1dfdbba1d812ab1918e0ce2884cdfdff0ceb0cb54bc56dc",
    ),
    (
        "Mekong_1396181",
        "train",
        2424141,
        "7441b4ce03633d3bb477d38383b9495081f80619644f4006c8cf0344fbd01be2",
        16633,
        "3cba6ca75f2c277d5e681437ebd8da0e45707e37b82c026b3bc799791ca16438",
    ),
    (
        "Mekong_16233",
        "train",
        2256144,
        "3022128b783154cacbae064f3b1c6c1aa7855d5c9dd91f47d61614a966b33c5e",
        4983,
        "fee52121a2abbd1e1f57f67927122530bf18b7eb336da7ecb2c137f5875f7102",
    ),
    (
        "Mekong_596495",
        "train",
        2411087,
        "7d2adab08af8d576203f17623a3860c6934d1f4079004701351588786f1eb383",
        11426,
        "9fbec9b1bd06e96cc8eb0c66fa4934a48dd80cbd1b670494c2affd259ee7a32c",
    ),
    (
        "Nigeria_529525",
        "train",
        2259657,
        "d84bb2914e73bdb952b820ad63c2c75363a85b2d0c5cf6ff8a5734d33abccc49",
        11861,
        "8a64830e6d1558ec6af8b52b6f4b1fe0ef8920d7f317cfdd8b37bac40f9e8fc9",
    ),
    (
        "Nigeria_600295",
        "train",
        2650879,
        "b418a36e2bd30dbb9452e64e30ec176258ea6a0691630ea980da9d705fa7c9c8",
        12927,
        "7bce532830162964a4d0283046affb74c09f2a1ce3ec5a3c32aa4a6fe2fc629f",
    ),
    (
        "Nigeria_952958",
        "train",
        2448466,
        "49c15c93e8c3e014b7423a6235b0811f83396878831fd23c3d4331611ec54af8",
        6941,
        "f1c8dc26d87e40a4be800f52e5eed855a815f2d1ffeaab476daa7dc729123af9",
    ),
    (
        "Pakistan_246510",
        "train",
        2334917,
        "4115017a9b122d7398822060ab7fe25e55d08543beaa24217ae394d6210eb2e4",
        3328,
        "2ecb22ece2e6ea5c462a9844f1a5f90b813f60366a76eab369b8779ae31be519",
    ),
    (
        "Pakistan_474121",
        "train",
        3172134,
        "37cf02dd20992ade67b071bdde11616a3b1f22ab49e0fb6ec4a8e255edae793c",
        6604,
        "33bae600a8c46dba840a953e338f9e35f4d29f86948cef7dbc20570a0a4966f7",
    ),
    (
        "Paraguay_126224",
        "train",
        1904691,
        "0801c5886ee0072707dd2aeadf99c4c5418a1c40784008cbdda46f1fd122b7fa",
        16766,
        "c8547c0d6031f040bd4e67804eb83339d17e1fd7753b8f18648e9f05f69aeaa8",
    ),
    (
        "Paraguay_822142",
        "train",
        2128900,
        "fdd18e61dfa59fc2c7b6d77823767e558fa84072b0d18f4fb25b8650a5b82c77",
        10588,
        "db2853aeda0cd616ebbe5a7bb5152e78425cd5edcf2f6db53d561f99cfd81d71",
    ),
    (
        "Somalia_1087508",
        "train",
        2579388,
        "d9a9a359407cdd14b96c8561cf9094fe1f02c05fc025bf77497793466797d4d9",
        10949,
        "869ec4bb24f5216b2092e6c8bb8a6a88a18244294212a4053d12771adf3741ee",
    ),
    (
        "Somalia_371421",
        "train",
        2603913,
        "da6858c7fe2d77313efbe849013c9f57c0ce503403929d4ec8d64960f8ca5717",
        12315,
        "f4f16eb1d1c0b74adc2e105ed35d6a457a5695c87980c6b017084fece1a6b9d5",
    ),
    (
        "Spain_2938657",
        "train",
        2364938,
        "ccc542052c18bcb9fc2c97cbc384df01075f30d07f3586761c9329bf22ee2476",
        1991,
        "9d9c2a1bdd688d7665388e8404832908a6b906bcc818f40743bb8ca267f7218f",
    ),
    (
        "Spain_8199661",
        "train",
        2316354,
        "e740ace73679f333f66ea88ae0c252daa030c0fb6489038adb54edb342deda31",
        3618,
        "73b55033d476fbe4744a1589e1cbb71632eef478732eb0819a1b0d8b09e2547b",
    ),
    (
        "Sri-Lanka_653336",
        "train",
        2322882,
        "1ec56dc78347e8e1d81db440ff494ec83d55f7f5b60e7669f6e50bb18a269968",
        5827,
        "a1b69316d33cc9fea53581aa50fcb9608d5fe4b385848e28ca8decd5d79397ad",
    ),
    (
        "Sri-Lanka_883641",
        "train",
        2243119,
        "4f155b3698cb493203762e3e51a0892a0f88909933d93cd174e8d4ffa5efec6a",
        1914,
        "ea7e06bf1badbf2779bd82da0ba9b37a611643d5ebeca1342617f7c525efa930",
    ),
    (
        "USA_1068362",
        "train",
        2131543,
        "82cca7cf72570d3fe2d8ce90c4b36ac326b3561d1dfee7741de98fceb9532f64",
        7009,
        "7cc38de430cf6908daa96a8800857f0b23c6aaa0377fbf04bcc5676f91dd15db",
    ),
    (
        "USA_652955",
        "train",
        2169702,
        "0cf186db362a8f4f414378b1083a25bb915c6ca90c80d126106082e4cc3078d4",
        2764,
        "3d40e98df00c30de3f84b577479c103750d64c5080f3aff7bc70178c798f2755",
    ),
    (
        "Ghana_868803",
        "validation",
        2321183,
        "f31eb46a7ad94dfb6a79cdfa4d6062839d0bd96db620debc64e83e2103d16401",
        7696,
        "b047d8f6d902b44a9c20dc6fce6dbced6b884b8260f1040e61b851050f0b9689",
    ),
    (
        "India_1068117",
        "validation",
        2450753,
        "6a59c927e6f0cabd3206e5303154e6cabe7c2d0669266ffe2649f56a0b39e7ff",
        15733,
        "db5284a5cd91eab5c70eb310350f4e77e861adef6d1daf26244fbfdd1b4c83e0",
    ),
    (
        "Mekong_293769",
        "validation",
        2448110,
        "4d9b10a1e4c7dc05a47d74397c334f25096acc9e1887659b3c0e2986ea6e6a98",
        6528,
        "629304ae517b2c7dd0dc6cb8dd72d5943b4a2943965bd782a7b309f28077ef73",
    ),
    (
        "Nigeria_984831",
        "validation",
        2362185,
        "a7d45a2f8ab4aa6fdd311fe25937e9112cb367fbf7a6cf306b69add29f20846b",
        10322,
        "99ac70c34aaca2ce48a157f247ef35e7eb1c57af3a58d5f6236e74b3156eed94",
    ),
    (
        "Pakistan_1027214",
        "validation",
        2245598,
        "c78cc47a8eeae63dfe57eb426dbe01c8d9b7f364d07889b0ce55d607869e2c85",
        10004,
        "82f1a0c7677d4b14ebda580884a6e8e854c2bf00ba5f470c439456e3da2c1902",
    ),
    (
        "Paraguay_581976",
        "validation",
        1914206,
        "aa0bf782e01fb60b002aa5ae2397d73fe07861d3ea8cc7a3aba9ffe7c00a6c5f",
        15614,
        "40035afa001ba84d7b59dd593fa2fa6743b1d4af1356b4f478046b734b870ad9",
    ),
    (
        "Somalia_12849",
        "validation",
        2617333,
        "43781ffe14ff9604d0538ec9a6f7c49ffb9ae488b0f08ffa3dcbbb1dc207b7d9",
        15222,
        "9b7cb1c8a378c193b1be9405c6ea200c38c3fc80300d190184c0226383c0afcb",
    ),
    (
        "Spain_1199913",
        "validation",
        2348104,
        "f76d9be2ab682c7c88fcaa3bf621aa19fa697fc82b848c1fe2034d0c36be5138",
        2348,
        "27fedadf6819805196eb1061a21e11ccb7ba26648a2424917b9de3174d45f2e0",
    ),
    (
        "Ghana_313799",
        "test",
        2243083,
        "621f5c19765e6e6a28d3a3b8bb357188ecb4d65ab4f14dba38e30bd84b93ce40",
        6073,
        "3bde81b5172d5017ec561ce137cdf8d59fbf3a69fc7796d08188b97219a32862",
    ),
    (
        "Ghana_319168",
        "test",
        2552380,
        "2a46104f1a75903ce2b0dd98b82e3e20236ea997071c6e4e801af903ad9d0445",
        11209,
        "da46da70d9b18f8851a58c3868a4a4b6262ebfa4433ff9306feaa101b634f33e",
    ),
    (
        "India_592446",
        "test",
        2418725,
        "4a041b3f3f28e5219da53c56d3715bc7a75925b235b69a5d1628951fd9e5805c",
        5708,
        "23cf1fcd9882fac00552bc25ec4202e95d58421eff597c9cf659c4f733c49cd0",
    ),
    (
        "India_747992",
        "test",
        2341250,
        "4e9cf7dff2a8b6495b18e4abf12cfe587a4b3d28deadccfe4354b60b1f40c9ae",
        17053,
        "c63a63987d65748a87ddd4c7685c7334eaa37955ce4e4cad6ce26af764bab8a9",
    ),
    (
        "Mekong_382276",
        "test",
        2268452,
        "3065622535054f19f5c517dcfe2308dea0a71825e2a86479d914be4932c9b31f",
        2201,
        "9ff60aa49161aec241227fd6e9fed2c89a6f22ec839ef9dbf76268d399996631",
    ),
    (
        "Nigeria_812045",
        "test",
        2433122,
        "14053b65f4f4d453c1cc7d7a4ff7b0a1124880f81404985d798ff616364699b6",
        9767,
        "77386f8586a6a825430b11d74619fd39391f2c5d3fc2fc93d829bc15f4933308",
    ),
    (
        "Pakistan_849790",
        "test",
        2179387,
        "facb36e9d7ad50bce80d8af8be7a95af3ad20c696b77896120db86d460cc8dce",
        17848,
        "669bcd99f5cedcf37db82c7eff7dc0c20036588eb1462321cf8fecf727945586",
    ),
    (
        "Paraguay_868895",
        "test",
        2511881,
        "a33f701211cd8a509e2c1428378d2e88039d162a8f1fcab68bf682882aa200e6",
        10893,
        "c4da12e7aa9eb2a42f3594c9744f63322b2ea6cdb0ce17963bfb114452e5d3c1",
    ),
    (
        "Somalia_94102",
        "test",
        2574036,
        "a7b40d8260e5e3f055c42e8e6c661f47fcb78a4b90b21d5522763ba335691b7a",
        6569,
        "5177ba119333f3d31a94c18f113bfe593925286dacbbb86780ff664b0ca49681",
    ),
    (
        "Spain_6095801",
        "test",
        2319727,
        "a21c5574a35f058bb3e3cefb1d6625b7a156bde8b38ad2b34ef09685863a01c9",
        7549,
        "f0c9a7ad9c7049ca43363580eb8c6849812b51bb8095901fcc2796400817f512",
    ),
    (
        "Sri-Lanka_377277",
        "test",
        2200577,
        "3caae5b0815a852bc6c8dae352c0906d13c2d63c1f04e8bc7d94537c94c4bf2d",
        3456,
        "6540085bae0fa58de10f7b6e0cdbb1df3498c2276b8c7b7f14628759c7cbc063",
    ),
    (
        "USA_430764",
        "test",
        2199014,
        "385418e105dc1068a7d78585e4395bcdc134e7b0d092ba942867ef74393d8d12",
        5944,
        "ae84a33d041a947fc20d36bdeb2176d80f50bc5161dafa4b372df9f9d75ebe50",
    ),
)
SAMPLE_LABEL_SOURCE = f"{CORPUS_NAME}; {CORPUS_RELEASE}; {CORPUS_LICENSE}"

# The pre-registered fresh test (2026-10-19): (chip, S2Hand bytes, S2Hand sha256, LabelHand bytes, LabelHand sha256).
FRESH_TEST_RECORDS: tuple[tuple[str, int, str, int, str], ...] = (
    (
        "India_1018327",
        2334123,
        "2d162175fec6842370d1bb224414f132dbc5463d0a0ef64baae8692942b5d585",
        11303,
        "98290650b74df82e3acc22aaef90bcb4272821b2afb53f05af683f94b5fe1b56",
    ),
    (
        "India_399883",
        2129825,
        "3af2d0b68c1615fbf3ccf93a1ceeaee50145810854c9bab14d8f9ed77c584079",
        9902,
        "2024277074e45016fa82d8300353d8ff1476dfe68eb72d6623b838b1d4dfe973",
    ),
    (
        "India_570384",
        2485522,
        "615bb89a7871fdb5bcb577e4cb3ad5a3b7ce581779d1d54a470fd36594fa57d3",
        9209,
        "15345107b9bef08c30951b22fe5240758bfbff1873daf55b6ff7c4ff9c83e96b",
    ),
    (
        "India_591317",
        2166164,
        "399e75dcc6b3bc268f427cb3aa70c47d828eb09eccb08f9ae8866b75c6cb3937",
        8323,
        "6f5f4c7a34bae3b62348af93b642b2e6e420c9a2d77a857145b46fda8348873a",
    ),
    (
        "India_828067",
        2650684,
        "77cb8feb1c094931f7bc23ca97d6fa12e2dd6cfe8c7d2e16c772350ee03af7b6",
        10782,
        "5f32ba153c9fa448b9ebdd8368bfff54b74c724a22ff644aaac1c55ace122237",
    ),
    (
        "India_952728",
        2612711,
        "55dca19c550154c560dcee7df23f7866eff90b96856e7c5f931f0d704c4a9b47",
        11069,
        "45edbf83b77d384bff6a15dd23d8a4e3ff5ddc3df28efe0704be843ca7b0583b",
    ),
    (
        "Mekong_1443339",
        2349634,
        "afbc01a747e5c6d7b6b51c733c92c187323f25d7a24f8f7411836a326b09201b",
        13991,
        "e306812aaf1587c46f9b2a0c63e4af4fe46ac9fd46a9e6286dd41d70f824454e",
    ),
    (
        "Pakistan_694942",
        2309701,
        "e6706858531cf83702787164c6b58125e55ce658cd28155c99fba14c7fccbb40",
        4661,
        "095b3e78ea2bed54ed45f453344afd9067fec33fc96b9e3ce8f9d144a4301323",
    ),
    (
        "Paraguay_1029191",
        2189228,
        "fc6a1bab4c554bb3370396e5412ff446302941dad1226ef6465fb3a1def39892",
        6080,
        "4186a68dae7bee570685f0c4879441ae5a9af6977e1097f215a685a098dd3cf5",
    ),
    (
        "Paraguay_232281",
        2009196,
        "b03027bb22e7940e541ec74b5a0f50a6a2c4d9a5d28cbe5610e3a017acab9fdd",
        6469,
        "ce0d15ffd97158c4340641268b31f7e73687c81ec3e2184c0a7d6881fd7e2f06",
    ),
    (
        "Paraguay_511199",
        2144324,
        "47dec4832e5ec24df44d6286f5251b7bbc00907ed2eb8340d2649f57a27045da",
        6187,
        "2ed431be505531a2147a4237d9d927ad89690b1a59e2b2d20d0b47ae70f2b0a8",
    ),
    (
        "Paraguay_651904",
        2090599,
        "180ab057b125584b7e83581e79be763b9e8b3368909ab388587b5f4119aa0fa9",
        3472,
        "44d1514eeef21b3a0c04aa0fa7c1c49ef780f4c4835c734bfa2db172b455d289",
    ),
    (
        "Paraguay_790830",
        2213654,
        "8bba3ef74cefe2f53d05043a5fd8dac50c1f94642b78ea5e58039706021d6189",
        3950,
        "256861e8f5e55203471d81b6dcdfe70c2ec9a915dc52eb3805fdd540f950a435",
    ),
    (
        "Somalia_322855",
        2319134,
        "908f9600f5d9537542656315f3dc1f79f5f6c92ca9fb72d53b2722f1d81e42af",
        11099,
        "c38f887de34306b7a40e8a0edbbc756f5dad289ee52c41f559cae9b29dcf4e43",
    ),
    (
        "Somalia_685158",
        2672633,
        "20d281d5cc8cafb963a89bdddbceb711e35b6b7784e64fcae38923db4d53283b",
        14360,
        "9a7bed202008b4ac9525d07c98c3d3882a9f1f121125df5b6c150731c9d6ff3a",
    ),
    (
        "Spain_5650136",
        2206996,
        "7d56099682c125ad85a5bcaac6623c0768ec3bb567dc8982ef3e8a4b36fc8a24",
        9758,
        "c0a53cf82bd45a0fb9b7f1c2e9408e9dd817c06a633045fce88b017b7aa990b5",
    ),
    (
        "Spain_7387658",
        2192267,
        "80b0abea99212e477d4925d04df7226b98d151218598631282dd6fb6d6da3d8d",
        19124,
        "d7f6c03457e5191733d46f9d52f8342af1d73cfbeafeeba100794443b01f79fd",
    ),
    (
        "Sri-Lanka_534068",
        1760466,
        "5aae329bce9537116dbff0de3d1fac40039a49f4b6c45447fe7a5eb62eff098c",
        1588,
        "e77e5e64025852b82ab5f8624ed126e866a6fff9c661635313aab9276522ad4b",
    ),
    (
        "Sri-Lanka_649970",
        2344414,
        "ea8a1788c248cfcd4217189cb67eefc30d0b3bc8235cd79b067b41fb70f743f6",
        8972,
        "d9ad0a6879b7a507e4030e7228ca021ee96666c15eb72e6b7700b7243ac5bc78",
    ),
    (
        "Sri-Lanka_713926",
        2361023,
        "40239d792db24b7c3027370699782b189fde8b10cd696422b17c68f596d7fcf6",
        4008,
        "b07e813683173d668958cb456048d9904ac9f7ed9e5c8d4791d93cdec04412cc",
    ),
    (
        "USA_595451",
        3030491,
        "0bef53ef00039674cde3ddc984d27c822a619c81467546be726c90126693b9be",
        6150,
        "5bd0328de97ba3435661ef47a313d96541234fb21240bba5e9f8c65f19b5317d",
    ),
    (
        "USA_67102",
        2163989,
        "d2f7395fcd935345d9435866ccb1c556c520c53ecf7acebd2325fac04cd0e98d",
        2413,
        "48c52abfe9c81476960a6a22924e5dbb42fe9016674ff13f3fa9e68693bcfe50",
    ),
    (
        "USA_905409",
        2093474,
        "e44a749f3d8cbbaebdb56c1284354040d7b965899dcdbb40454f5c4ba46c60eb",
        3954,
        "dfb9d85d04740b0f4555043171b581fef080ea989120c6d43694def8df416de1",
    ),
)
FRESH_TEST_EVENTS = ("India", "Mekong", "Pakistan", "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA")
FRESH_TEST_EXCLUDED = ("India_900498", "Spain_7370579")  # upstream example chips the notebook predicts and shows
FRESH_POOL_DIGEST = "b7706fad689c52b3b848e2bed19da276d59d00cc187f693a0ce249f229ef4815"
FRESH_TEST_BYTES = 53_017_076


def fresh_pool_digest(rows: Sequence[Sequence[Any]] = FRESH_TEST_RECORDS) -> str:
    """SHA-256 of the compact JSON of the sorted pin rows (the pre-registered fresh-pool identity)."""
    canonical = json.dumps(sorted([list(r) for r in rows]), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_path(name: str, kind: str) -> str:
    """Bucket-relative path of a chip's image (`S2Hand`) or label (`LabelHand`) object."""
    folder = {"image": "S2Hand", "label": "LabelHand"}[kind]
    return f"data/flood_events/HandLabeled/{folder}/{name}_{folder}.tif"


def fetch_object(rel: str, *, expected: tuple[int, str], cache_dir: str | Path | None = None, fetcher: Any = None) -> bytes:
    """Return one pinned object's bytes from the cache or the bucket, refused on a size or digest mismatch."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / Path(rel).name
    size, digest = expected
    data = local.read_bytes() if local.is_file() else b""
    if len(data) != size or _sha256_bytes(data) != digest:
        url = CORPUS_BASE_URL + rel
        if fetcher is not None:
            data = fetcher(url)
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "dimer-prithvi-flood-tutorial/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (pinned https URL)
                data = response.read()
        if len(data) != size or _sha256_bytes(data) != digest:
            raise ValueError(
                f"{rel}: fetched {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, pinned {size} / {digest[:16]}…"
            )
        local.write_bytes(data)
    return data


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, dict[str, bytes]]:
    """Every pinned chip's image and label bytes, keyed by chip name."""
    out = {}
    for name, _role, image_bytes, image_sha, label_bytes, label_sha in SAMPLE_RECORDS:
        out[name] = {
            "image": fetch_object(
                object_path(name, "image"), expected=(image_bytes, image_sha), cache_dir=cache_dir, fetcher=fetcher
            ),
            "label": fetch_object(
                object_path(name, "label"), expected=(label_bytes, label_sha), cache_dir=cache_dir, fetcher=fetcher
            ),
        }
    return out


def read_corpus(files: Mapping[str, Mapping[str, bytes]]) -> dict[str, list[dict[str, Any]]]:
    """Decode the verified bytes into `{id, image, label}` records grouped by role (train / validation / test)."""
    import tempfile

    splits: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    for name, role, *_ in SAMPLE_RECORDS:
        if name not in files:
            raise ValueError(f"corpus is missing {name}")
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.tif"
            label_path = Path(tmp) / "label.tif"
            image_path.write_bytes(files[name]["image"])
            label_path.write_bytes(files[name]["label"])
            image = read_chip(image_path, band_indices=S2_L1C_BAND_INDICES)
            label = read_mask(label_path)
        raw = {
            "id": f"{role}-{len(splits[role]):03d}",
            "source_id": name,
            "region": name.split("_")[0],
            "split": role,
            "image": image,
            "label": label,
            "source": f"{CORPUS_BASE_URL}{object_path(name, 'image')}",
        }
        splits[role].append(check_record(raw))  # scaled to reflectance, no-data replaced, label checked
    return splits


def fetch_fresh_test(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> list[dict[str, Any]]:
    """The 23 pre-registered fresh-test chips as checked records (role 'fresh'), each object refused on a size or
    digest mismatch. Call it only where the fresh test may be scored (the notebook's fresh-test section)."""
    import tempfile

    if fresh_pool_digest() != FRESH_POOL_DIGEST:
        raise ValueError("FRESH_TEST_RECORDS do not match the pre-registered fresh-pool digest")
    records = []
    for i, (name, image_bytes, image_sha, label_bytes, label_sha) in enumerate(FRESH_TEST_RECORDS):
        image = fetch_object(object_path(name, "image"), expected=(image_bytes, image_sha), cache_dir=cache_dir, fetcher=fetcher)
        label = fetch_object(object_path(name, "label"), expected=(label_bytes, label_sha), cache_dir=cache_dir, fetcher=fetcher)
        with tempfile.TemporaryDirectory() as tmp:
            image_path, label_path = Path(tmp) / "image.tif", Path(tmp) / "label.tif"
            image_path.write_bytes(image)
            label_path.write_bytes(label)
            raw = {
                "id": f"fresh-{i:03d}",
                "source_id": name,
                "region": name.split("_")[0],
                "split": "fresh",
                "image": read_chip(image_path, band_indices=S2_L1C_BAND_INDICES),
                "label": read_mask(label_path),
                "source": f"{CORPUS_BASE_URL}{object_path(name, 'image')}",
            }
        records.append(check_record(raw))
    return records


def fresh_test_metadata(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> list[dict[str, Any]]:
    """Resolve and parse every pinned fresh chip WITHOUT scoring: fetch (size + digest verified) and read the TIFF
    headers only (shape, dtype); no pixel is decoded and no model is touched."""
    import tifffile

    out = []
    for name, image_bytes, image_sha, label_bytes, label_sha in FRESH_TEST_RECORDS:
        entry: dict[str, Any] = {"chip": name}
        for kind, expected in (("image", (image_bytes, image_sha)), ("label", (label_bytes, label_sha))):
            data = fetch_object(object_path(name, kind), expected=expected, cache_dir=cache_dir, fetcher=fetcher)
            with tifffile.TiffFile(io.BytesIO(data)) as tif:
                series = tif.series[0]
                entry[kind] = {"shape": list(series.shape), "dtype": str(series.dtype)}
        out.append(entry)
    return out


def fetch_sample_dataset(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus (roles from the official Sen1Floods11 splits)."""
    return read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no chip (by pixel digest) appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = chip_digest(record)
            if key in seen and seen[key] != name:
                raise ValueError(f"chip {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train / validation / test after de-duplicating chips. Chips from one
    scene or event are near-duplicates; group them yourself (one region per split) when that matters."""
    import random

    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = chip_digest(record)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_test = max(1, round(len(unique) * test_fraction))
    n_val = round(len(unique) * val_fraction)
    splits = {"test": unique[:n_test], "validation": unique[n_test : n_test + n_val], "train": unique[n_test + n_val :]}
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(f"split leaves {len(splits['train'])} training chips; at least {MIN_RECORDS} are required")
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, image, label}` records from a directory or a zip holding `pairs.csv` (columns `id`, `image`,
    `label`) beside six-band 512 × 512 GeoTIFF chips and single-band label rasters; files are decoded from bytes,
    never extracted to disk."""
    import tempfile

    source = Path(path)
    if source.is_dir():
        table = (source / "pairs.csv").read_text(encoding="utf-8")
        loader = lambda name: (source / name).read_bytes()  # noqa: E731
    elif source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        members = {Path(n).name: n for n in archive.namelist()}
        if "pairs.csv" not in members:
            raise ValueError("BYOD zip must contain pairs.csv")
        table = archive.read(members["pairs.csv"]).decode("utf-8")
        loader = lambda name: archive.read(members[name])  # noqa: E731
    else:
        raise ValueError("BYOD datasets must be a directory or a .zip holding pairs.csv and the GeoTIFF files")
    rows = list(csv.DictReader(io.StringIO(table)))
    missing = {"id", "image", "label"} - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"pairs.csv is missing columns {sorted(missing)}")
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for row in rows:
            image_path = Path(tmp) / "image.tif"
            image_path.write_bytes(loader(row["image"]))
            record: dict[str, Any] = {"id": row["id"], "image": read_chip(image_path)}
            if row.get("label"):
                label_path = Path(tmp) / "label.tif"
                label_path.write_bytes(loader(row["label"]))
                record["label"] = read_mask(label_path)
            out.append(record)
    return out


def write_sample_pair(record: Mapping[str, Any], image_path: str | Path, label_path: str | Path) -> dict[str, str]:
    """Write one record as a six-band float32 TIFF and a single-band int16 TIFF (the BYOD shape, without
    georeferencing) and return both paths."""
    import numpy as np
    import tifffile

    image_out, label_out = Path(image_path), Path(label_path)
    image_out.parent.mkdir(parents=True, exist_ok=True)
    image_buf, label_buf = io.BytesIO(), io.BytesIO()
    tifffile.imwrite(image_buf, np.asarray(record["image"], dtype=np.float32), photometric="minisblack", planarconfig="separate")
    tifffile.imwrite(label_buf, np.asarray(record["label"], dtype=np.int16), photometric="minisblack")
    atomic_write_bytes(image_out, image_buf.getvalue())  # R70-m6: never a half-written file at the final path
    atomic_write_bytes(label_out, label_buf.getvalue())
    return {"image": str(image_out), "label": str(label_out)}


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the pairs table of a split (id, image, label, provenance) in the shape BYOD expects."""
    out = Path(path)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=["id", "image", "label", "region", "source"])
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "id": record["id"],
                "image": f"{record.get('source_id', record['id'])}_S2Hand.tif",
                "label": f"{record.get('source_id', record['id'])}_LabelHand.tif",
                "region": record.get("region", ""),
                "source": record.get("source", ""),
            }
        )
    atomic_write_bytes(out, handle.getvalue().encode("utf-8"))
    return out


def dataset_manifest(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Validate every split and summarise the dataset (counts, class balance, digests) for provenance exports."""
    summary: dict[str, Any] = {"model_id": MODEL_ID, "image_size": IMAGE_SIZE, "splits": {}}
    for name, records in splits.items():
        report = validate_dataset(records, min_records=1)
        summary["splits"][name] = {
            "n_records": report["n_records"],
            "class_pixel_fraction": report["class_pixel_fraction"],
            "ignored_pixels": report["ignored_pixels"],
            "regions": sorted({str(r.get("region", "")) for r in records if r.get("region")}),
            "digest": report["digest"],
        }
    summary["disjoint"] = check_split_disjoint(splits)
    digests = json.dumps({k: v["digest"] for k, v in summary["splits"].items()}, sort_keys=True)
    summary["digest"] = hashlib.sha256(digests.encode("utf-8")).hexdigest()
    return summary
