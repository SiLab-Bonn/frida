# Usage

ADC capacitor stages, comparator decisions, and output codes follow the
[project-wide ADC convention](adc_conventions.md): C0 is the largest and first
capacitor, B0..B16 are chronological decisions, and DOUT is the 12-bit result
formed from all 17 decisions.

FRIDA commands run through the Python module that owns the operation:

```text
python -m flow.<block>.primitive
python -m flow.<block>.sim
python -m flow.util.netlist
python -m flow.scans.<scan>
```

There is no installed `flow` executable or separate build-system orchestration
layer. Run commands from the repository root through `uv`, for example:

```bash
uv run python -m flow.comp.sim --help
uv run python -m flow.comp.sim
```

Digital lint, simulation, synthesis, and implementation use their stock tools
directly.

## Layout primitives

The MOSFET and MOM-capacitor generators are directly executable modules:

```bash
uv run python -m flow.mosfet.primitive \
  [-t <tech>] [-m <mode>] [-v] [-o <dir>]

uv run python -m flow.momcap.primitive \
  [-t <tech>] [-m <mode>] [-v] [-o <dir>]
```

The process-independent FRIDA capacitor-array runners are:

```bash
uv run python -m flow.cdac.layout
uv run python -m flow.cdac.layout caparray_1layer_radix17
uv run python -m flow.cdac.layout caparray_2layer_radix17
uv run python -m flow.cdac.layout caparray_3layer_radix17
```

Each target visibly constructs its electrical `CdacParams`, unit family, and
metal stack. The generator derives every physical distance from the selected
PDK, builds the complete `1..coarse_weight` unit family, partitions arbitrary
positive electrical weights through the shared netlist decomposition, and
adds routing, shield taps, connection stacks, and pins. A named target always
runs `gdscheck`, foundry Calibre DRC/LVS, and xACT PEX. Results are written to
`build/layout/cdac/<target>/<timestamp>/`.

ADC layout validation and extraction use the same named-target convention as
simulation. Omitting the target lists the four archived FRIDA-1 ADCs and two
generated FRIDA-2 stacks:

```bash
uv run python -m flow.adc.layout
uv run python -m flow.adc.layout frida1_1layer_radix17
uv run python -m flow.adc.layout frida1_2layer_radix20
uv run python -m flow.adc.layout frida2_2layer_radix17
uv run python -m flow.adc.layout frida2_3layer_radix17
```

A target always generates and runs the complete signoff sequence. The DRC
stage includes the PDK-local `gdscheck` ADC suite before foundry Calibre DRC.
Results are isolated beneath `build/layout/adc/<target>/<timestamp>/`. Each
result contains the exact GDS and source CDL used, a recognition-only GDS diff
where applicable, generated decks and logs, DRC/LVS reports, coupling and
net-summary reports, and the final RC-plus-coupling `*.pex.netlist`. xACT stops
unless its own conductive-source LVS comparison is correct.

For FRIDA-1, the run-local layout copy adds only PDK-local non-mask MOM
recognition shapes; the archived source and GDS are never modified. The normal
LVS source contains the intended per-layer capacitor network. Thus the old
one-layer layout must report `CORRECT`, while the known disconnected upper
layer in an old two-layer layout must report the expected connectivity warning.
The separate conductive PEX source omits the electrically empty CDAC wrappers
so xACT extracts the fabricated connectivity.

The FRIDA-2 targets regenerate their CDAC GDS and ideal per-chunk MOM source
network from the target's `CdacParams` and PDK rules, then perform strict block
substitution in `build/frida-2-template.gds`. The assembler requires identical
database units, boundaries, pin names, pin positions, and pin shapes. It never
edits, moves, or reroutes layout; any required template adjustment is made
manually in KLayout. The M5-M7 caparray itself contains no M3 and uses M4 as a
partitioned shared layer: 100 nm plate routes occupy the two edge corridors
while the shield remains beneath the central capacitor body.

| Option | Values | Default |
|---|---|---|
| `-t, --tech` | `ihp130`, `tsmc65`, `tsmc28`, `tower180` | `ihp130` |
| `-m, --mode` | `min`, `max` | `min` |
| `-v, --visual` | render the generated GDS | off |
| `-o, --out` | output directory | `build` |

```bash
uv run python -m flow.mosfet.primitive -t ihp130 -m max -v
```

## Circuit simulation targets

ADC, comparator, sampler, and CDAC simulations use the same named-target
interface. Omitting the target prints the choices without generating or
running anything:

```bash
uv run python -m flow.adc.sim
uv run python -m flow.comp.sim
uv run python -m flow.samp.sim
uv run python -m flow.cdac.sim
```

Each target owns its reviewed parameter recipe. Short, noise-free Spectre
diagnostics use the same Python target with `check=True`, through pytest only:

```bash
source /eda/local/scripts/cadence_2024-25.sh
uv run pytest -m spectre
# Select just one worker campaign's preflight, if needed:
uv run pytest -m spectre -k 'adc-frida2_fixed_input_noise'
```

Allow roughly one hour for the full Spectre suite on asiclab003; runtime depends
on the host and available resources. Run the full group occasionally or after
broad simulation-flow changes, not after every edit. For routine changes, use
`-k` to select the tests covering the affected block or experiment. "Short"
describes the simulated time, not the wall-clock runtime of extracted circuits.

The `spectre` group is excluded by default and requires no custom pytest flag.
It covers all eleven experiment targets, including six representative comparator
cases. Artifacts and diagnostic reports remain beneath timestamped
`build/diagnostics/` directories. Missing Spectre or an unavailable PDK installation
skips with a reason; missing configured inputs, invalid decks, license failures,
and simulator errors fail. Circuit-check findings remain available in reports;
these diagnostics are not comprehensive circuit signoff. There are no CLI
diagnostic flags, netlist-only modes, or separate check targets.

## Circuit simulation

Simulation targets use the same interface and always create a fresh complete
run beneath `build/sim/<module>/<target>/<YYYYMMDD_HHMMSS>/`:

```bash
# Reviewed ADC campaigns
uv run python -m flow.adc.sim hdl21_fixed_input_noise_vs_rate
uv run python -m flow.adc.sim frida1_fixed_input_noise_vs_rate
uv run python -m flow.adc.sim frida1_fixed_input_noise
uv run python -m flow.adc.sim frida2_fixed_input_noise
uv run python -m flow.adc.sim hdl21_transfer_curve
uv run python -m flow.adc.sim frida1_transfer_curve

# Comparator campaigns
uv run python -m flow.comp.sim hdl21_comp_perf_vs_size
uv run python -m flow.comp.sim frida1_fixed_input_noise

# First standalone block-level simulations
uv run python -m flow.samp.sim frida1_transient
uv run python -m flow.cdac.sim frida1_transfer_curve
```

The extracted fixed-input rate sweep groups the one-layer and two-layer radix-17
and radix-20 ADCs beneath one timestamped run. Each flavor has its own named
subdirectory containing the 2, 6, and 10 MS/s results. Independent cases run
concurrently within a fixed worker budget, while retaining one coherent
campaign directory.

The ADC and comparator runners convert completed raw results to typed HDF5 in
each case directory. Sampler and CDAC retain the Spectre raw result and log;
their standalone runners do not yet invoke the existing typed measurement and
analysis path. The comparator analysis validates the 297 typed HDF5 files
and their embedded candidate metadata directly, without a separate campaign
manifest. Accepted analysis directories remain explicit paths in
`flow.analysis.runner` and are updated manually after reviewing a run.

Every executable target builds one native HDL21 `Sim` per parameter variant
and calls `Sim.run()` or `hs.run()` with the timestamped output as the VLSIR
run directory. ADC and comparator batches use isolated spawned processes.
The comparator campaign runs 24 workers with one Spectre thread each; the
single-case sampler and CDAC targets run directly. ADC and comparator convert
the returned transient to typed HDF5 before releasing it; the standalone
sampler and CDAC targets retain the native raw result without converting it.
VLSIR retains the generated `netlist.scs` in every run directory, so a separate
netlist-writing layer or target is unnecessary.

Concurrency and each case's Spectre `+mt` setting are visible in the target
which owns the campaign. Simulation targets require `spectre` on `PATH`; source
`design/spice/workspace.sh` first when necessary.

## Netlist conversion

Netlist utilities are subcommands of their owning module:

```bash
uv run python -m flow.util.netlist oa-to-cdl \
  --cdslib cadence/cds.lib --lib frida --cell core \
  --outdir build/netlist

uv run python -m flow.util.netlist cdl-to-sp \
  design/spice/core.cdl build/netlist/core.sp

uv run python -m flow.util.netlist clean-cdl \
  design/spice/core.cdl build/netlist/core.sp \
  --verilog design/hdl/frida_core.v --module frida_core
```

`clean-cdl` removes filler and decap instances and normalizes OpenROAD
hierarchy names. Pass `--verilog` and `--module` together to reorder the
subcircuit ports using a Verilog module declaration.

## Digital checks

The normal software test suite includes both SPI-register implementations as
cocotb tests. It uses Icarus by default:

```bash
uv run pytest
uv run pytest -q -s test/test_spi_register.py
```

Use another cocotb-supported simulator through `SIM`:

```bash
SIM=verilator uv run pytest -q -s test/test_spi_register.py
```

The simulator is a system dependency, not a Python package. For Ubuntu:

```bash
sudo apt install iverilog
```

Verible, Verilator, Yosys, and OpenROAD can likewise be run directly for
linting, synthesis, and physical implementation. The OpenROAD configuration is
under `design/`; see [`openroad.md`](openroad.md) for project-specific notes.

## ADC scans

Hardware acquisition uses explicit targets in the physical scan runner:

```bash
uv run python -m flow.scans.runner --help
```

The explicit ADC00--ADC03 slow-ramp campaign uses the same acquisition path:

```bash
uv run python -m flow.scans.runner adc_ramp_code_density
```

It records one four-million-conversion sawtooth capture per ADC for transfer,
code-density, DNL, and INL analysis. The stored DOUT is retained as the
uncalibrated result; BOUT can also be decoded in memory with the measured P/N
CDAC weights for the switching direction selected by each element's programmed
A-state. The A-state determines the direction of any physical change: an
element initially at zero can only rise, while one initially at one can only
fall. BOUT separately selects the final states imposed by the SAR logic:
`P_final = 1 - BOUT` and `N_final = BOUT`. Therefore BOUT does not always mean
"move P" or "move N". For one capacitor pair, the calibrated weight is the
distance between its two possible BOUT endpoints, equal to the sum of its
direction-matched P and N movements. The `adc_ramp_nonlinearity` analysis runner
performs this decoding.

```bash
uv run python -m flow.analysis.runner adc_ramp_nonlinearity
```

The maintained runner pins one reviewed ADC00--ADC03 ramp directory and the
reviewed CDAC campaign directories. It validates transport counters and capture
lengths, excludes eight conversions after each detected sawtooth flyback,
requires both decodings to be monotonic within 2 LSB, and writes the plots plus
one consolidated metrics CSV below a fresh `build/analysis/adc/` directory.

The digital-calibration flow compares three ways to derive backend BOUT weights
for ADC00. `calibration1.py` uses the direction-matched physical CDAC S-curves,
`calibration2.py` performs a nonnegative fit against the known ramp, and
`calibration3.py` extracts the all-zero/all-one prefix thresholds described by
Hsu. None changes the analog ADC. The two ramp-derived methods use disjoint
training and validation cycles, and the threshold method preserves nominal
ratios once the measured steps become noise-limited.

```bash
uv run python -m flow.analysis.runner adc_calibration
```

All three analyses return the same typed 17-weight result. The runner writes a
shared weight comparison, transfer, code-density, and INL/DNL plots plus metrics
and normalized-weight CSV files below one fresh `build/analysis/adc/` directory.
Every weight vector sums to 4095; rounding to a 12-bit integer is deferred until
the final backend output.

Run one explicitly named physical campaign through the shared scan runner:

```bash
uv run python -m flow.scans.runner adc_sine_conversion_rate
uv run python -m flow.scans.runner adc_fixed_input_noise_50mv_700mvcm
uv run python -m flow.scans.runner adc_ramp_code_density
uv run python -m flow.scans.runner comp_common_mode
uv run python -m flow.scans.runner cdac_cap_mismatch
```

Use `--help` to list every maintained ADC, comparator, CDAC, and repair target.
The target function owns the complete parameter recipe. ADC targets iterate
their flat list and pass one configuration plus its lifecycle position to the
acquisition module; comparator and CDAC targets pass their complete lists.
Each run writes one typed HDF5
measurement per parameter variant below a fresh timestamped `build/scan_adc/`,
`build/scan_comp/`, or `build/scan_cdac/` directory. The individual scan modules
are libraries and do not provide command-line entry points.

Behavioral and SPICE-backed scans use the same acquisition schema. The ADC
Spectre flow exposes one fixed-input noise campaign for each DUT view:

```bash
uv run python -m flow.scans.scan_behavioral
uv run python -m flow.adc.sim hdl21_fixed_input_noise_vs_rate
uv run python -m flow.adc.sim frida1_fixed_input_noise_vs_rate
```

Use the `spectre` pytest group for short diagnostics with circuit checks and
AHDL linting. Normal CLI runs write below a fresh
`build/sim/adc/<target>/<YYYYMMDD_HHMMSS>/`; omitting the target lists all choices.

For the extracted-layout comparison, `frida1_fixed_input_noise` selects all four
historical flavors and `frida2_fixed_input_noise` selects the connected one-,
two-, and three-layer radix-17 variants. Each case uses 100 conversions, 50 mV
differential input, 700 mV common mode, 1.2 V supplies, and transient device noise. These reuse
the same testbench and result conversion as the rate-sweep targets.

```bash
uv run pytest -m spectre -k 'adc and fixed_input_noise and not vs_rate'
uv run python -m flow.adc.sim frida1_fixed_input_noise
uv run python -m flow.adc.sim frida2_fixed_input_noise
uv run python -m flow.analysis.runner adc_pex_flavor_paths --inputs /path/to/completed/campaign
```

Worker preflight uses these diagnostic tests: it checks the actual extracted
port order and internal waveform nodes, generates the complete input, and runs
short Spectre diagnostics. Require passing tests, not skipped prerequisites,
before deploying a long campaign.
Each case records its PEX SHA-256. A worker snapshot must include the selected
`build/layout/adc/<target>/<timestamp>/` signoff summary and PEX netlist.
Only the two historical two-layer targets may accept the explicitly recorded
disconnect warning; FRIDA-2 requires raw LVS `CORRECT`.

These comparison targets pin their reviewed extraction directories explicitly;
they do not search for the latest extraction. Update those paths deliberately
after reviewing a replacement signoff run. `_run_adc_sim()` executes one case;
the named targets own the experiment parameters and concurrent submission.
Targets are listed in HDL21, FRIDA-1, then FRIDA-2 order, without old-name aliases.

The standard sequence has a 100 ns active conversion window at a 1.6 GHz
symbol rate, followed by padding to a 160 ns record. Thus “10 MS/s” describes
the active timing, not a continuous 100 ns sampling interval. The analysis
uses all 100 decision records and writes both B0-first 17-bit decisions and
decoded 12-bit outputs. Run long campaigns in detached worker sessions and
copy results back before selecting them for analysis.

## Environment setup

Clone with submodules, create the environment, and run the software checks:

```bash
git clone --recursive git@github.com:kcaisley/frida.git
cd frida
uv sync
uv run pytest
```

Cadence Spectre, ngspice, Xyce, Icarus, KLayout, and OpenROAD are external
executables. Install only the tools needed for the workflows you run and make
them available on `PATH`.
