##  python3.7 exp4.py | grep -v -e "commercial" -e "Username"

import warnings

warnings.filterwarnings("ignore")
import os
import time
import utils
import pandas as pd
import gc
import argparse
from multiprocessing import Pool, Value, cpu_count, Process

from ACCESS2020.ACCESS2020 import ACCESS2020
from ASPDAC2022.ASPDAC2022 import ASPDAC2022
from CIE2021.CIE2021 import CIE2021
from COR2022.COR2022 import COR2022
from IEEEJAS2021.IEEEJAS2021 import IEEEJAS2021
from IEEETII2020.IEEETII2020 import IEEETII2020
from RTAS2018.RTAS2018 import RTAS2018
from RTAS2020.RTAS2020 import RTAS2020
from RTCSA2018.RTCSA2018 import RTCSA2018
from RTCSA2020.RTCSA2020 import RTCSA2020
from RTNS2016.RTNS2016 import RTNS2016
from RTNS2016_nowait.RTNS2016_nowait import RTNS2016_nowait
from RTNS2017.RTNS2017 import RTNS2017
from RTNS2021.RTNS2021 import RTNS2021
from RTNS2022.RTNS2022 import RTNS2022
from SIGBED2019.SIGBED2019 import SIGBED2019
from GLOBECOM2022.GLOBECOM2022 import GLOBECOM2022

FUNC = {
    "SIGBED2019": SIGBED2019,
    "COR2022": COR2022,
    "CIE2021": CIE2021,
    "RTNS2017": RTNS2017,
    "RTNS2016": RTNS2016,
    "RTNS2016_nowait": RTNS2016_nowait,
    "RTNS2021": RTNS2021,
    "RTNS2022": RTNS2022,
    "ASPDAC2022": ASPDAC2022,
    "IEEETII2020": IEEETII2020,
    "RTCSA2018": RTCSA2018,
    "IEEEJAS2021": IEEEJAS2021,
    "ACCESS2020": ACCESS2020,
    "GLOBECOM2022": GLOBECOM2022,
    "RTCSA2020": RTCSA2020,
    "RTAS2018": RTAS2018,
    "RTAS2020": RTAS2020,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="GLOBECOM2022")
    parser.add_argument("--ins", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--path", type=str, default=f"../data/grid/0/")

    args = parser.parse_args()

    ins = args.ins
    EXP = "grid"
    directory = pd.read_csv(args.path + "dataset_logs.csv")[
        args.start : args.end
    ]

    DATA = args.path + "{}_task.csv" 
    TOPO = args.path + "{}_topo.csv" 

    for name, method in [
        (name, method)
        for name, method in FUNC.items()
        if name in args.method or args.method.lower() == "all"
    ]:
        utils.init(exp=EXP, ins=ins, method=name)
        print("---------------%s-%s-----------------" % (name, EXP))

        signal = Value("i", 1000)
        oom = Process(
            target=utils.killif,
            args=(
                os.getpid(),
                utils.process_num(name),
                utils.t_limit,
                signal,
            ),
        )
        oom.start()

        utils.rheader()

        result_log = open(f"result_{ins}_{name}.csv", "w")
        result_log.write(
            ",".join(["piid", "is_feasible", "solve_time", "total_time", "total_mem"])
            + "\n"
        )

        def write_result(result):
            result_log.write(result + "\n")
            result_log.flush()

        with Pool(
            processes=cpu_count() // utils.process_num(name), maxtasksperchild=1
        ) as p:
            for i, row in directory.iterrows():
                piid = row["id"]
                p.apply_async(
                    method,
                    args=(
                        DATA.format(piid),
                        TOPO.format(piid),
                        piid,
                        f"../configs/{EXP}/{ins}/{name}/",
                        utils.process_num(name),
                    ),
                    callback=write_result,
                )

            p.close()
            try:
                while signal.value > 0:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Terminate calculation by hand.")

        oom.terminate()
        gc.collect()
        time.sleep(1)
