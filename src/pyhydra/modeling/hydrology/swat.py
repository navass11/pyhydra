"""
SWAT+ automation utilities.

Handles climate input file generation and scenario execution for SWAT+.

Requires:
    - SWAT+ executable (Rev. 60.5.4 or compatible).
    - pandas, numpy.
    - pySWATPlus (calibration only): ``pip install pySWATPlus``
    - spotpy (calibration only): ``pip install spotpy``
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def write_precipitation_file(
    df_coords: pd.DataFrame,
    df_series: pd.DataFrame,
    output_path: str,
    missing_value: float = -99.0,
) -> None:
    """Write a SWAT+ precipitation input file (.pcp format).

    Args:
        df_coords: DataFrame with columns ['Station', 'Lati', 'Long', 'Elev'],
                   index = station IDs.
        df_series: DataFrame with a datetime index and one column per station
                   (same order as df_coords). Values in mm/day.
        output_path: Output file path (e.g. 'TxtInOut/pcp1.pcp').
        missing_value: Value used for missing data (-99.0 by default).
    """
    stations = list(df_coords["Station"])
    lines: list[str] = [
        "Observed precipitation (mm)\n",
        "nbyr   tstep   lat        lon        elev\n",
    ]
    for _, row in df_coords.iterrows():
        lines.append(f"    0       0  {row['Lati']:9.4f}  {row['Long']:9.4f}  {row['Elev']:9.1f}\n")

    df_filled = df_series[stations].fillna(missing_value)
    for date, row in df_filled.iterrows():
        values = "".join(f"{v:8.2f}" for v in row)
        lines.append(f"{date.strftime('%Y%j')}{values}\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("".join(lines))
    print(f"✓ Precipitation file written → {output_path}")


def write_temperature_file(
    df_coords: pd.DataFrame,
    df_tmax: pd.DataFrame,
    df_tmin: pd.DataFrame,
    output_path: str,
    missing_value: float = -99.0,
) -> None:
    """Write a SWAT+ temperature input file (.tmp format).

    Args:
        df_coords: DataFrame with columns ['Station', 'Lati', 'Long', 'Elev'].
        df_tmax: Daily maximum temperature (°C), same structure as df_tmin.
        df_tmin: Daily minimum temperature (°C).
        output_path: Output file path (e.g. 'TxtInOut/tmp1.tmp').
        missing_value: Value used for missing data.
    """
    stations = list(df_coords["Station"])
    lines: list[str] = [
        "Observed temperature (°C)\n",
        "nbyr   tstep   lat        lon        elev\n",
    ]
    for _, row in df_coords.iterrows():
        lines.append(f"    0       0  {row['Lati']:9.4f}  {row['Long']:9.4f}  {row['Elev']:9.1f}\n")

    tmax = df_tmax[stations].fillna(missing_value)
    tmin = df_tmin[stations].fillna(missing_value)
    for date in tmax.index:
        row_max = tmax.loc[date]
        row_min = tmin.loc[date]
        values = "".join(f"{mx:8.2f}{mn:8.2f}" for mx, mn in zip(row_max, row_min))
        lines.append(f"{date.strftime('%Y%j')}{values}\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("".join(lines))
    print(f"✓ Temperature file written → {output_path}")


def edit_file_cio(file_cio_path: str, start_year: int, end_year: int) -> None:
    """Set the SWAT+ simulation period.

    SWAT+ rev 60+ (SWAT+ Editor projects) delegates the simulation period to
    ``time.sim`` — ``file.cio`` in these projects only lists per-category
    config filenames and has no "IYR"/"NBYR" fields. This function locates
    ``time.sim`` next to the given ``file.cio`` and rewrites its single data
    row (``day_start yrc_start day_end yrc_end step``). The parameter name is
    kept as ``file_cio_path`` for backward compatibility with existing call
    sites that point at ``file.cio``.

    Args:
        file_cio_path: Path to ``file.cio`` (``time.sim`` is expected in the
            same directory, as written by the SWAT+ Editor).
        start_year: First calendar year of the simulation.
        end_year: Last calendar year of the simulation.

    Raises:
        FileNotFoundError: If ``time.sim`` does not exist next to
            ``file_cio_path``.
        ValueError: If ``time.sim`` does not have the expected 3-line
            header + data-row layout.
    """
    time_sim_path = Path(file_cio_path).with_name("time.sim")
    if not time_sim_path.exists():
        raise FileNotFoundError(
            f"time.sim not found next to {file_cio_path} "
            "(SWAT+ rev 60+ stores the simulation period there, not in file.cio)"
        )

    lines = time_sim_path.read_text().splitlines(keepends=True)
    if len(lines) < 3:
        raise ValueError(f"Unexpected time.sim format in {time_sim_path}")

    lines[2] = f"{0:7d}{start_year:11d}{364:10d}{end_year:9d}{0:10d}\n"
    time_sim_path.write_text("".join(lines))
    print(f"✓ time.sim updated: yrc_start={start_year}, yrc_end={end_year}.")


def write_swatplus_precipitation_files(
    df_stations: pd.DataFrame,
    df_series: pd.DataFrame,
    txtinout_dir: str,
    missing_value: float = -99.0,
) -> None:
    """Write SWAT+ individual precipitation files (one .pcp file per station).

    SWAT+ uses one file per station (e.g. p1.pcp, p2.pcp) referenced from
    pcp.cli, instead of the legacy SWAT 2012 multi-station format.

    Args:
        df_stations: DataFrame with columns ['name', 'lat', 'lon', 'elev'].
                     Each row defines one precipitation station.
        df_series: DataFrame with a datetime index and one column per station
                   (column names must match df_stations['name']). Values in mm/day.
        txtinout_dir: Path to the SWAT+ TxtInOut directory.
        missing_value: Value written for missing data (-99.0 by default).
    """
    out_dir = Path(txtinout_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    for _, row in df_stations.iterrows():
        name = row["name"]
        fname = f"{name}.pcp"
        filenames.append(fname)
        series = df_series[name].fillna(missing_value)
        nbyr = int(series.index.year.max() - series.index.year.min() + 1) if len(series) else 0
        lines = [
            "Precipitation data - file written by pyhydra\n",
            f"nbyr     tstep       lat       lon      elev\n",
            f"{nbyr:4d}         0  {float(row['lat']):9.3f}  {float(row['lon']):9.3f}  {float(row['elev']):9.3f}\n",
        ]
        for date, val in series.items():
            lines.append(f"{date.year:4d}  {date.dayofyear:3d}  {val:10.5f}\n")
        (out_dir / fname).write_text("".join(lines))

    cli_lines = ["pcp.cli: Precipitation station files - written by pyhydra\nfilename\n"]
    cli_lines += [f"{fn}\n" for fn in filenames]
    (out_dir / "pcp.cli").write_text("".join(cli_lines))
    print(f"✓ {len(filenames)} SWAT+ precipitation files written → {out_dir}")


def write_swatplus_temperature_files(
    df_stations: pd.DataFrame,
    df_tmax: pd.DataFrame,
    df_tmin: pd.DataFrame,
    txtinout_dir: str,
    missing_value: float = -99.0,
) -> None:
    """Write SWAT+ individual temperature files (one .tmp file per station).

    Args:
        df_stations: DataFrame with columns ['name', 'lat', 'lon', 'elev'].
        df_tmax: Daily max temperature (°C), columns matching df_stations['name'].
        df_tmin: Daily min temperature (°C), same structure.
        txtinout_dir: Path to the SWAT+ TxtInOut directory.
        missing_value: Value written for missing data.
    """
    out_dir = Path(txtinout_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    for _, row in df_stations.iterrows():
        name = row["name"]
        fname = f"{name}.tmp"
        filenames.append(fname)
        tmax = df_tmax[name].fillna(missing_value)
        tmin = df_tmin[name].fillna(missing_value)
        nbyr = int(tmax.index.year.max() - tmax.index.year.min() + 1) if len(tmax) else 0
        lines = [
            "Temperature data - file written by pyhydra\n",
            f"nbyr     tstep       lat       lon      elev\n",
            f"{nbyr:4d}         0  {float(row['lat']):9.3f}  {float(row['lon']):9.3f}  {float(row['elev']):9.3f}\n",
        ]
        for date in tmax.index:
            lines.append(f"{date.year:4d}  {date.dayofyear:3d}  {tmax[date]:10.5f}  {tmin[date]:10.5f}\n")
        (out_dir / fname).write_text("".join(lines))

    cli_lines = ["tmp.cli: Temperature station files - written by pyhydra\nfilename\n"]
    cli_lines += [f"{fn}\n" for fn in filenames]
    (out_dir / "tmp.cli").write_text("".join(cli_lines))
    print(f"✓ {len(filenames)} SWAT+ temperature files written → {out_dir}")


def read_channel_sd(
    output_dir: str,
    freq: str = "day",
    unit: int | None = None,
) -> pd.DataFrame:
    """Read a SWAT+ ``channel_sd_<freq>.txt`` output file.

    SWAT+ writes per-channel flow, sediment and nutrient balances to
    ``channel_sd_day.txt`` / ``_mon.txt`` / ``_yr.txt`` / ``_aa.txt`` in the
    TxtInOut / scenario directory. This reader handles the fixed layout
    written by SWAT+ (title line, whitespace-separated column-name line,
    units line, then data), including the unnamed flow-rate summary columns
    that SWAT+ repeats as ``null`` in the header (m3/s, one per storage/
    inflow/outflow block) — renamed here to ``null``, ``null_1``, ``null_2``
    so every column has a unique name.

    Args:
        output_dir: TxtInOut / scenario directory containing the file.
        freq: One of ``'day'``, ``'mon'``, ``'yr'``, ``'aa'`` (average annual).
        unit: Channel unit id to filter to (SWAT+ ``unit`` column). If None,
            all channels are returned.

    Returns:
        DataFrame indexed by date for ``'day'``/``'mon'``/``'yr'`` (last day
        of the period), or by integer position for ``'aa'`` (no meaningful
        date — one row per channel, averaged over the whole simulation).
        Columns are the SWAT+ output variables (``flo_out``, ``sed_out``,
        ``no3_out``, ...).

    Raises:
        ValueError: If ``freq`` is not one of the supported values.
        FileNotFoundError: If the output file does not exist.
    """
    if freq not in ("day", "mon", "yr", "aa"):
        raise ValueError(f"Unknown freq '{freq}'. Use 'day', 'mon', 'yr' or 'aa'.")

    path = Path(output_dir) / f"channel_sd_{freq}.txt"
    if not path.exists():
        raise FileNotFoundError(f"SWAT+ output file not found: {path}")

    with open(path) as fh:
        fh.readline()  # title
        raw_cols = fh.readline().split()

    seen: dict[str, int] = {}
    col_names = []
    for name in raw_cols:
        if name in seen:
            seen[name] += 1
            col_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            col_names.append(name)

    df = pd.read_csv(path, sep=r"\s+", skiprows=3, names=col_names, header=None)

    if unit is not None:
        df = df[df["unit"] == unit].copy()

    if freq == "day":
        date = pd.to_datetime(df["yr"].astype(str) + "-01-01") + pd.to_timedelta(
            df["jday"] - 1, unit="D"
        )
        df = df.set_index(date).rename_axis("date")
    elif freq == "mon":
        date = pd.to_datetime(
            df["yr"].astype(str) + "-" + df["mon"].astype(str).str.zfill(2) + "-01"
        ) + pd.offsets.MonthEnd(0)
        df = df.set_index(date).rename_axis("date")
    elif freq == "yr":
        date = pd.to_datetime(df["yr"].astype(str) + "-12-31")
        df = df.set_index(date).rename_axis("date")
    # freq == "aa": no meaningful date, keep default integer index.

    return df


def read_swat_discharge(output_dir: str, unit: int, freq: str = "day") -> pd.Series:
    """Read simulated outlet discharge (``flo_out``, m3/s) for one channel.

    Convenience wrapper around :func:`read_channel_sd` for the common case of
    extracting a single channel's discharge time series.

    Args:
        output_dir: TxtInOut / scenario directory containing the SWAT+ output.
        unit: Channel unit id (SWAT+ ``unit`` column, e.g. the outlet reach).
        freq: One of ``'day'``, ``'mon'``, ``'yr'``.

    Returns:
        pd.Series of ``flo_out`` (m3/s) indexed by date, named ``'flo_out'``.
    """
    if freq == "aa":
        raise ValueError("read_swat_discharge requires a dated freq: 'day', 'mon' or 'yr'.")
    df = read_channel_sd(output_dir, freq=freq, unit=unit)
    return df["flo_out"].rename("flo_out")


def run_swat(model_dir: str, swat_exe: str, timeout: int = 3600) -> int:
    """Execute the SWAT+ model in the given directory.

    Args:
        model_dir: Path to the TxtInOut directory (where swat.exe is run from).
        swat_exe: Path to the SWAT+ executable.
        timeout: Maximum run time in seconds (default 3600 s = 1 h).

    Returns:
        Process return code (0 = success).
    """
    result = subprocess.run(
        [swat_exe],
        cwd=model_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode == 0:
        print(f"✓ SWAT+ finished successfully in {model_dir}.")
    else:
        print(f"✗ SWAT+ failed (returncode={result.returncode}).")
        print(result.stderr[-2000:])
    return result.returncode


# ── Calibration helper ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SWATCalibrationParameter:
    """Definition of one SWAT+ calibration parameter (pySWATPlus convention).

    Attributes:
        name: SWAT+ parameter name, e.g. ``"cn2"``, ``"esco"``.
        change_type: pySWATPlus change type — ``"absval"`` (replace the
            parameter value), ``"pctchg"`` (percent change) or ``"relchg"``
            (relative change).
        lower: Lower bound for SCE-UA search.
        upper: Upper bound for SCE-UA search.
    """

    name: str
    change_type: str
    lower: float
    upper: float


class SWATModel:
    """Calibratable SWAT+ model backed by pySWATPlus.

    Applies candidate parameter changes via ``pySWATPlus.TxtinoutReader``,
    runs SWAT+, and reads back outlet discharge with :func:`read_channel_sd`.

    Args:
        txtinout_dir: Path to the reference SWAT+ TxtInOut project used as the
            calibration template (read by pySWATPlus; each run is materialised
            under ``sim_dir``, the template itself is not modified in place).
        sim_dir: Directory where each simulation run is written and executed.
        parameters: Calibration parameter definitions.
        outlet_unit: SWAT+ channel ``unit`` id for the calibration outlet. If
            None, auto-detected after the first run as the maximum ``unit`` id
            in ``channel_sd_<freq>.txt`` (LREW-style single-outlet convention).
        freq: :func:`read_channel_sd` frequency (``'day'``, ``'mon'``, ``'yr'``).
        flow_col: Output column with simulated discharge (default ``'flo_out'``).
    """

    def __init__(
        self,
        txtinout_dir: str,
        sim_dir: str,
        parameters: list[SWATCalibrationParameter],
        outlet_unit: int | None = None,
        freq: str = "mon",
        flow_col: str = "flo_out",
    ):
        self.txtinout_dir = str(txtinout_dir)
        self.sim_dir = str(sim_dir)
        self.parameters = parameters
        self.outlet_unit = outlet_unit
        self.freq = freq
        self.flow_col = flow_col
        self._reader = None

    @property
    def parameter_bounds(self) -> list[tuple[str, float, float]]:
        return [(p.name, p.lower, p.upper) for p in self.parameters]

    def _get_reader(self):
        if self._reader is None:
            try:
                from pySWATPlus import TxtinoutReader
            except ImportError as exc:
                raise ImportError("SWATModel requires pySWATPlus: pip install pySWATPlus") from exc
            self._reader = TxtinoutReader(self.txtinout_dir)
        return self._reader

    def run_swat(self, *params) -> pd.Series:
        """Apply calibration parameters, run SWAT+, and return simulated discharge.

        Args:
            *params: Parameter values in the order given by ``parameters``.

        Returns:
            Series of ``flow_col`` indexed by date (from :func:`read_channel_sd`),
            restricted to ``outlet_unit``.
        """
        if len(params) != len(self.parameters):
            raise ValueError(f"Expected {len(self.parameters)} SWAT+ parameters, received {len(params)}")

        param_dicts = [
            {"name": p.name, "change_type": p.change_type, "value": float(v)}
            for p, v in zip(self.parameters, params)
        ]
        reader = self._get_reader()
        reader.run_swat(sim_dir=self.sim_dir, parameters=param_dicts)

        df = read_channel_sd(self.sim_dir, freq=self.freq)
        if df.empty:
            raise RuntimeError(f"No SWAT+ output rows found in {self.sim_dir} (channel_sd_{self.freq}.txt)")
        if self.outlet_unit is None:
            self.outlet_unit = int(df["unit"].max())
        df = df[df["unit"] == self.outlet_unit]
        if df.empty or self.flow_col not in df.columns:
            raise RuntimeError(f"No SWAT+ output values found in {self.sim_dir} for unit {self.outlet_unit}")

        return df[self.flow_col].rename("sim").sort_index()


def _align_swat_series(simulation, evaluation) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(simulation, pd.Series) and isinstance(evaluation, pd.Series):
        idx = simulation.dropna().index.intersection(evaluation.dropna().index)
        if len(idx):
            return simulation.loc[idx].to_numpy(dtype=float), evaluation.loc[idx].to_numpy(dtype=float)
    sim = np.asarray(simulation, dtype=float)
    obs = np.asarray(evaluation, dtype=float)
    n = min(len(sim), len(obs))
    return sim[:n], obs[:n]


def validate_swat_parameter_sensitivity(
    model: SWATModel,
    baseline: np.ndarray | None = None,
    perturbed: np.ndarray | None = None,
    min_delta: float = 1e-5,
) -> dict[str, float]:
    """Run baseline and perturbed SWAT+ simulations and confirm parameters matter.

    Unlike HEC-HMS calibration parameters (multiplicative factors around a
    baseline), SWAT+ parameters use absolute search ranges, so the default
    baseline is the midpoint of each parameter's bounds rather than 1.0.
    """
    if baseline is None:
        baseline = np.array([(p.lower + p.upper) / 2.0 for p in model.parameters], dtype=float)
    if perturbed is None:
        perturbed = np.array(
            [p.lower if i % 2 == 0 else p.upper for i, p in enumerate(model.parameters)],
            dtype=float,
        )
    q0 = model.run_swat(*baseline)
    q1 = model.run_swat(*perturbed)
    idx = q0.index.intersection(q1.index)
    if len(idx) == 0:
        raise RuntimeError("SWAT+ sensitivity validation failed: runs do not overlap.")
    max_delta = float(np.nanmax(np.abs(q0.loc[idx].values - q1.loc[idx].values)))
    if not np.isfinite(max_delta) or max_delta < min_delta:
        raise RuntimeError("SWAT+ parameter edits did not change the hydrograph; calibration stopped.")
    return {
        "baseline_peak": float(q0.loc[idx].max()),
        "perturbed_peak": float(q1.loc[idx].max()),
        "max_delta": max_delta,
    }


def calibrate_swat_sceua(
    model: SWATModel,
    observed: pd.Series,
    dbname: str,
    n_evals: int = 100,
    objective: str = "nse",
    ngs: int | None = None,
):
    """Calibrate a SWAT+ model with SPOTPY SCE-UA.

    A failed simulation (SWAT+ crash, pySWATPlus error) is treated as a very
    poor fit rather than aborting the whole search: the inner ``simulation``
    call catches exceptions from :meth:`SWATModel.run_swat` and substitutes
    an all-zero series, which the objective function then scores as a very
    bad NSE/RMSE — letting SCE-UA continue exploring instead of crashing.

    Args:
        model: Calibratable SWAT+ model.
        observed: Observed discharge series, at the same frequency as
            ``model.freq``. A ``pd.Series`` with a ``DatetimeIndex`` is
            aligned by date; a plain array-like is aligned by position.
        dbname: SPOTPY database path without ``.csv``.
        n_evals: Number of SCE-UA evaluations.
        objective: ``"nse"`` (minimize ``-NSE``), ``"pbias_abs"``, or ``"rmse"``.
        ngs: Number of SCE-UA complexes. Defaults to ``n_parameters + 1``.

    Returns:
        The SPOTPY sampler after completion.
    """
    try:
        import spotpy
    except ImportError as exc:
        raise ImportError("calibrate_swat_sceua requires spotpy") from exc

    observed = observed if isinstance(observed, pd.Series) else pd.Series(observed)
    observed = observed.dropna()

    class _Setup:
        def __init__(self):
            self.observed = observed
            self.params = [
                spotpy.parameter.Uniform(name, lower, upper)
                for name, lower, upper in model.parameter_bounds
            ]

        def parameters(self):
            return spotpy.parameter.generate(self.params)

        def simulation(self, vector):
            try:
                return model.run_swat(*vector)
            except Exception as exc:
                print(f"  simulation error: {exc}")
                return np.zeros(len(self.observed))

        def evaluation(self):
            return self.observed

        def objectivefunction(self, simulation, evaluation):
            sim, obs = _align_swat_series(simulation, evaluation)
            if len(sim) == 0:
                return np.inf
            if objective == "nse":
                return -spotpy.objectivefunctions.nashsutcliffe(obs, sim)
            if objective == "pbias_abs":
                return abs(spotpy.objectivefunctions.pbias(obs, sim))
            return spotpy.objectivefunctions.rmse(obs, sim)

    sampler = spotpy.algorithms.sceua(_Setup(), dbname=dbname, dbformat="csv")
    sampler.sample(n_evals, ngs=ngs or len(model.parameters) + 1)
    return sampler
