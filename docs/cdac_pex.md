# Caparray capacitance comparison

## C0-first full-ADC restart (2026-09-05)

The FRIDA-2 runners now read the paired `build/frida-2-template-c0.gds` and
`build/frida-2-template-c0.cdl`. The annotation-only migration and before/after
conductive LVS checks are recorded beneath
`build/layout/adc/template_annotations/20260905_190843/`. Original files were
preserved. Geometry and instance placements did not change during renaming.

Assembly checks the terminal-name set, database unit, and footprint, leaving
electrical connectivity to full-chip LVS. Marker coordinates/layers and all
generated caparray polygons are instead pinned by the ordinary regression
test in `pdk/tsmc65/tests/test_cdac_layout.py`. Its baseline comes from the
human-reviewed, signed-off September 5 arrays, not the historical template.

All three assembled FRIDA-2 variants pass raw MOM-aware LVS, the configured
DRC checks with zero errors, and xACT PEX:

- `build/layout/adc/frida2_1layer_radix17/20260905_193440/`
- `build/layout/adc/frida2_2layer_radix17/20260905_193629/`
- `build/layout/adc/frida2_3layer_radix17/20260905_193816/`

The simulation view `frida2` uses the renamed C0-first PEX ports and source
nets. `frida65a` retains the fabricated FRIDA-1 namespace and its explicit
15-first mapping. Both are checked against actual PEX headers and device
terminal names before simulation. The four historical inputs remain the
September 5 17:12 extractions; their two-layer raw LVS warnings represent the
known disconnected upper plates, not a repaired model of fabricated silicon.

The restart retains 100 conversions per flavor, 50 mV differential input,
700 mV common mode, 1.2 V supplies, and transient noise. The active conversion
is 100 ns within the intentionally retained 160 ns padded record. All sixteen
main/diff state and driver nodes on both branches are saved, together with the
comparator cross-coupled nodes and both input-pair drains. Additional traces
are stored in HDF5 `wave/internal_v/`; the existing plotter-facing traces stay
in `wave/`. `info/readbacks/signal_map_json` records their exact PEX node names.
No initial-state voltages or physical connections are changed by renaming.

The `frida1_fixed_input_noise` and `frida2_fixed_input_noise` family runners
launch independent flavors concurrently, with separate spawned Python processes
for HDL21/PDK state isolation and one named output subdirectory per flavor. FRIDA-1 uses
four six-thread Spectre jobs; FRIDA-2 uses three eight-thread jobs. Both
budgets total 24 compute threads on a 28-physical-core worker. Input manifests
record the per-case thread count. Netlist-only preflights exercise the same
process scheduling, and a failed case propagates failure to the family runner.
These are explicit runner budgets, not dynamic license/CPU allocation.

## Shared HDL21 source (2026-09-05)

`flow.cdac.subckt.CdacArray` now generates both the ideal and LVS capacitor
arrays. Both views have the same physical chunk/layer instances and the same
ordered 34-port interface: topplate, shield, main C0..C15, diff C0..C15.
The handwritten CDAC and ADC LVS circuit builders have been removed.
Only the leaf view changes: `flow.momcap.subckt.MomCap` contains an ideal
capacitor with an unused shield terminal; `pdk.tsmc65.signoff.mom_lvs_device`
supplies the three-terminal recognition model for Calibre.

`CdacParams.unit_cap` retains its meaning as the complete stack's logical
unit capacitance. Its ideal value is divided equally over the selected
layers, preserving the requested main/diff/effective totals. This is an ideal
model convention, not a claim of equal physical layer contributions. LVS
checks connectivity; the current recognition deck does not compare numerical
capacitance. PEX remains the source of actual capacitance and shunt loading.

The generic netlist replacement utility reconnects callers by formal pin
name, rather than copying positional arguments. The FRIDA-1 runner explicitly
maps fabricated driver pin 15 to C0; the renamed FRIDA-2 source uses C0 directly.
Calibre's raw PEX header may still have its own ordering; identical
ideal/LVS source ordering does not imply normalized PEX ordering.

Fresh standalone signoff results are in
`build/layout/cdac/main/20260905_133303/`, with one subdirectory per profile.
All three pass the existing DRC selections, raw LVS and xACT PEX. Their
geometric XOR against the reviewed GDS below is empty on every layer, and
capacitances are unchanged. `shared_source_verification.json` records the
matching source headers, unchanged reference/template hashes, and three
negative LVS cases that still fail (inner plate, topplate, and shield).
Full-ADC radix-17 checks also preserve FRIDA-1's results: the one-layer run
`build/layout/adc/frida1_1layer_radix17/20260905_133416/` is raw LVS `CORRECT`;
the two-layer run `build/layout/adc/frida1_2layer_radix17/20260905_133418/`
still reports only the known `vdac_p`/`vdac_n` disconnects with matching device
and port counts. Both complete conductive-source PEX. These checks do not
resolve the outstanding strict FRIDA-2 template interface mismatch.

## Reviewed interface revision (2026-09-05)

The current generator incorporates the geometry confirmed in the KLayout
review. All coordinates below are local to the caparray. The footprint is
18.400 x 53.080 um, with 41 functional chunks and two complete edge dummies.
The dummy inner plates float intentionally; their active-layer outer rings
join the signal topplate and their shield-layer metal joins the lower shield.

M4 landings are 0.100 x 0.520 um on 0.400 um unit pitch. Multiunit groups use
full-height horizontal connection bars. Lower landings span Y=0.22..0.74 and
upper landings Y=52.06..52.58. The M4-M7 plate contacts use two 0.100 um cuts
on 0.200 um pitch; the right-side topplate uses a 2x2 array. These dimensions
are derived from the selected PDK rules, including minimum metal area, rather
than embedded distances in the generic generator. The shortest finger now
accommodates one complete minimum-area landing, replacing the earlier
five-track tail assumption.

The three-layer profile uses M4 for both routing and the lower shield, with
M5-M7 active. Its shield spans Y=3.22..49.70. Sixteen 0.100 um horizontal
bridges extend to X=18.40 as VSS abutment tabs. Their pitch is 3.00 um within
each half and 4.38 um between the central pair. The existing 0.380 um bent
M6 topplate output is preserved. There is no M3, M8 or M9 geometry.

All three standalone profiles pass `gdscheck`, Calibre DRC with the existing
block-level exclusions (`PO.DN.2`, `DRM.R.1`, `MOM.R.1`), raw LVS `CORRECT`,
and the XRC-aware LVS used for xACT extraction. Fresh capacitances per array
side are:

| Active stack | Main (fF) | Diff (fF) | Main + diff (fF) | Effective (fF) | Top-shield (fF) |
|---|---:|---:|---:|---:|---:|
| M6, M5 shield | 327.457 | 43.892 | 371.349 | 283.565 | 97.518 |
| M6-M7, M5 shield | 616.439 | 82.898 | 699.337 | 533.542 | 95.202 |
| M5-M7, M4 shield | 891.398 | 119.898 | 1011.296 | 771.500 | 89.662 |

Results, including GDS, source CDL, reports, manifests and per-stage tables,
are under `build/layout/cdac/`:

- `caparray_1layer_radix17/20260905_123045/`
- `caparray_2layer_radix17/20260905_123044/`
- `caparray_3layer_radix17/20260905_122920/`

The three-layer directory's `five_design_capacitance.csv` and `.json` contain
the 34-row, five-design comparison in C0-first order, reusing the September 3
historical full-ADC extractions for the two FRIDA-1 columns.

The three-layer result also contains `diff_reference.gds` and
`diff_reference.json`, comparing against `caparray_2layer_radix17` inside
`build/frida-1.gds`. The bounding box, all 32 bottom-plate pin boxes and their
M4 connection bars match. On M6 the only differences are the removal of eight
obsolete shield-tap landing islands. This does not imply strict ADC interface
compatibility: pin labels and the topplate/shield pin layers still differ.
Neither the historical GDS nor the user-edited ADC template was modified.
Fault-injection copies in `negative_lvs/` independently remove the two-cut
inner-plate connection, the four-cut upper topplate connection, or all sixteen
shield tabs. Each produces raw LVS `INCORRECT`; `review_verification.json`
records these checks. Removing just one redundant cut is not a disconnect.

The September 3 results below describe the earlier geometry and are retained
as historical measurements; they are not the capacitances of this revision.

## Five-design radix-17 comparison (2026-09-03)

The refactored generator was extracted with the same TSMC65
`Calibre/rcx/calibre.rcx` xACT configuration as the earlier OA run: an
XRC-aware LVS database, capacitance-only PDB extraction, and a lumped Spectre
netlist using source net names. These values are for one standalone physical
cap-array side. A complete differential ADC contains one such array on each
side; the two sides should not be blindly added when referring input-referred
sampling noise.

`Main` and `diff` are the sums of every topplate-to-corresponding-bottom-plate
coupling over the 16 conversion stages. `Main + diff` is the physical
bottom-plate loading;
`effective` is `main - diff`, the signal-producing switched term. `Non-CDAC`
is `topplate total - main - diff`. In the standalone arrays it is exactly the
identified topplate-to-shield capacitance. In a complete historical ADC it
also includes small couplings to other top-level nets.

| Design | Main (fF) | Diff (fF) | Main + diff (fF) | Effective (fF) | Top-shield (fF) | Non-CDAC (fF) |
|---|---:|---:|---:|---:|---:|---:|
| FRIDA-1 1-layer radix-17 | 305.686 | 41.325 | 347.011 | 264.361 | 159.428 | 162.733 |
| FRIDA-1 disconnected 2-layer radix-17 | 306.105 | 41.424 | 347.529 | 264.681 | 79.706 | 83.011 |
| New M6 1-layer array | 341.685 | 48.663 | 390.348 | 293.022 | 63.799 | 63.799 |
| New connected M6-M7 2-layer array | 596.130 | 85.309 | 681.439 | 510.820 | 62.835 | 62.835 |
| New connected M5-M7 3-layer array | 890.786 | 125.559 | 1016.345 | 765.227 | 59.239 | 59.239 |

The connected second and third layers therefore do contribute materially, but
not as exact 2x/3x copies. Vertical coupling, coupling between active layers,
the fixed lower shield, and fixed access routing make the increment sublinear.
Importantly, the topplate-to-shield term stays approximately fixed at 59-64 fF
rather than doubling or tripling with the active-layer count.

All three profiles pass the `gdscheck` precheck and the full foundry Calibre DRC
with zero errors. Both signoff LVS and the XRC-aware LVS report `CORRECT`.
Removing an inner-plate via, removing the topplate via, or removing the complete
repeated shield-tap connection makes LVS fail as intended. No generated profile
contains M8 or M9 geometry.

The fresh historical 1-layer full-ADC run also reports raw LVS `CORRECT`. The
historical 2-layer run reports the expected raw LVS `INCORRECT`: all device and
port counts match, but only `vdac_p` and `vdac_n` have connectivity errors.
Its separate conductive-source xACT LVS reports `CORRECT`, allowing extraction
of the fabricated disconnected structure while retaining the explicit warning.

The complete five-design result bundle is in
`build/cdac_pex/20260903_165847`. Its root
`five_design_capacitance.csv`/`.json` files provide the requested 34 rows: 16
main values, 16 diff values, their summed total, and non-CDAC shunt loading.
`capacitance_summary.json` also records the effective values. Each new profile
directory contains its GDS, CDL, geometry manifest, per-stage capacitance table,
DRC/LVS reports, xACT PEX netlist, and logs. The two FRIDA-1 directories contain
fresh full-ADC extractions made by the same named workflow.

The generated 2-layer profile is compared directly against
`caparray_2layer_radix17` in `build/frida-2-template.gds` by
`diff_historical_2layer_radix17.gds`; datatype 0 is common, datatype 1 is
historical-only, and datatype 2 is generated-only on each source layer shifted
by +1000. The old cell is 18.400 x 53.080 um; the standalone rule-derived cell
is 16.740 x 53.760 um. Its conductors and rings are 100 nm wide, their ordinary
lateral gap is 100 nm, and the distinct dense line-end gap is 120 nm. The
400 nm unit pitch, 400 nm weight step, and 1 um residual tail all follow from
the PDK rules and dimensionless topology parameters. The standalone X width
omits the historical edge dummies. The strict ADC assembler currently rejects
this interface difference. It does not apply a Python geometry repair; the
template must be adjusted manually in KLayout before full FRIDA-2 ADC
extraction.

Compared with the old full-ADC 1-layer extraction below, the new standalone M6
macro has 293.022 fF rather than 264.361 fF effective capacitance. That 10.8%
difference includes the corrected rule-derived geometry and the fact
that the new number is a standalone macro extraction, whereas the historical
number was read through the routed `VDAC_P`/`VDAC_N` nets of the full ADC.

## Historical fabricated-array comparison

The following table compares the intended behavioral capacitor values in
`caparray.cdl` / `caparray.sp` against the physical-layout extraction for
`adc_1layer_radix17`.

Definitions:

- Intended: behavioral CDL model values from `caparray.cdl`.
- Layout P-side: current layout-derived PEX values from the `VDAC_P` branch of `adc_1layer_radix17.pex.netlist`.
- Extracted avg: average of the P-side and N-side PEX values.
- Eff: effective switched capacitance, `main - diff`.

The fresh PEX files used were generated in `/users/kcaisley/asiclab/tech/tsmc65/cds/PEX` on 2026-06-08 around 22:48.

| Stage | Weight | Intended main (fF) | Intended diff (fF) | Intended eff (fF) | Layout P main (fF) | Layout P diff (fF) | Layout P eff (fF) | Extracted avg main (fF) | Extracted avg diff (fF) | Extracted avg eff (fF) | Eff ratio ext/int |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 768 | 619.200 | 4.800 | 614.400 | 100.419 | 1.147 | 99.272 | 100.419 | 1.147 | 99.272 | 0.162 |
| C1 | 512 | 412.800 | 3.200 | 409.600 | 66.946 | 0.769 | 66.177 | 66.946 | 0.769 | 66.177 | 0.162 |
| C2 | 320 | 258.000 | 2.000 | 256.000 | 41.839 | 0.483 | 41.356 | 41.840 | 0.483 | 41.356 | 0.162 |
| C3 | 192 | 154.800 | 1.200 | 153.600 | 25.102 | 0.292 | 24.810 | 25.103 | 0.292 | 24.810 | 0.162 |
| C4 | 96 | 90.400 | 13.600 | 76.800 | 14.660 | 2.269 | 12.390 | 14.654 | 2.269 | 12.385 | 0.161 |
| C5 | 64 | 51.600 | 0.400 | 51.200 | 8.366 | 0.097 | 8.268 | 8.362 | 0.097 | 8.264 | 0.161 |
| C6 | 32 | 38.800 | 13.200 | 25.600 | 6.291 | 2.174 | 4.117 | 6.288 | 2.174 | 4.114 | 0.161 |
| C7 | 24 | 35.600 | 16.400 | 19.200 | 5.775 | 2.726 | 3.049 | 5.772 | 2.725 | 3.047 | 0.159 |
| C8 | 12 | 30.800 | 21.200 | 9.600 | 5.000 | 3.472 | 1.528 | 4.997 | 3.471 | 1.526 | 0.159 |
| C9 | 10 | 30.000 | 22.000 | 8.000 | 4.873 | 3.591 | 1.282 | 4.870 | 3.590 | 1.280 | 0.160 |
| C10 | 5 | 28.000 | 24.000 | 4.000 | 4.549 | 3.916 | 0.632 | 4.546 | 3.914 | 0.632 | 0.158 |
| C11 | 4 | 27.600 | 24.400 | 3.200 | 4.484 | 3.979 | 0.506 | 4.482 | 3.977 | 0.505 | 0.158 |
| C12 | 4 | 27.600 | 24.400 | 3.200 | 4.484 | 3.980 | 0.504 | 4.482 | 3.978 | 0.504 | 0.157 |
| C13 | 2 | 26.800 | 25.200 | 1.600 | 4.354 | 4.106 | 0.248 | 4.352 | 4.105 | 0.247 | 0.154 |
| C14 | 1 | 26.400 | 25.600 | 0.800 | 4.289 | 4.168 | 0.121 | 4.287 | 4.167 | 0.121 | 0.151 |
| C15 | 1 | 26.400 | 25.600 | 0.800 | 4.289 | 4.168 | 0.121 | 4.287 | 4.167 | 0.121 | 0.151 |
| Sum | 2047 | 1884.800 | 247.200 | 1637.600 | 305.720 | 41.340 | 264.380 | 305.686 | 41.324 | 264.361 | 0.161 |

## Notes

- The extracted `main - diff` relative weights match the intended `CAP_WEIGHTS` well after normalization.
- The absolute effective capacitance scale is about `16.1%` of the intended behavioral CDL value, or about `6.2x`
  smaller.
- The current physical result is consistent with the second layer in the 2-layer experiment being disconnected rather
  than contributing a parallel capacitance.

## How the extracted values were read

The extracted values in this table come from the fresh xACT-style PEX netlist:

```text
/users/kcaisley/asiclab/tech/tsmc65/cds/PEX/adc_1layer_radix17.pex.netlist
```

Do not use the stale side files for this run unless their timestamps show they were regenerated:

```text
adc_1layer_radix17.pex.netlist.ADC_1LAYER_RADIX17.pxi
adc_1layer_radix17.pex.netlist.pex
```

For the 2026-06-08 xACT/report run, the updated file was the top-level `adc_1layer_radix17.pex.netlist`, which contains
lumped capacitor lines like:

```spice
C12345 VDAC_P:17 DAC_DRIVE_BOTPLATE_MAIN_P<15>:42 0.123f
```

The topplate formal pin `cap_topplate` is flattened/promoted to the top-level nets:

```text
XXcaparray_p/cap_topplate -> VDAC_P
XXcaparray_n/cap_topplate -> VDAC_N
```

The historical botplate pins are the top-level drive nets:

```text
DAC_DRIVE_BOTPLATE_MAIN_P<physical-pin>
DAC_DRIVE_BOTPLATE_DIFF_P<physical-pin>
DAC_DRIVE_BOTPLATE_MAIN_N<physical-pin>
DAC_DRIVE_BOTPLATE_DIFF_N<physical-pin>
```

FRIDA-1 uses the legacy boundary mapping `stage = 15 - physical-pin`, so pin
15 is C0 and pin 0 is C15. The value for each table entry is the sum of all
capacitors between the topplate and corresponding botplate, including
colon-suffixed subnodes. For example:

```text
C0 main P = sum Cx where one terminal matches VDAC_P or VDAC_P:<subnode>
                     and the other matches DAC_DRIVE_BOTPLATE_MAIN_P<15>
                     or DAC_DRIVE_BOTPLATE_MAIN_P<15>:<subnode>
```

The same rule is used for `diff`, and for the `N` branch with `VDAC_N`.

Manual search in `coupling_capacitance.report` is useful for spot-checking large entries, for example:

```text
DAC_DRIVE_BOTPLATE_MAIN_P<15>  VDAC_P
```

but it is not sufficient for the complete table, because the report can omit smaller logical pair couplings. The full
PEX netlist should be parsed and summed.

The calculation is automated by `flow.cdac.pex`. Historical ADC extraction
must call `parse_adc_cdac_pex(..., stage_count=16,
pin_order="frida1_legacy")`; newly generated arrays use the default direct
stage order. Both return arrays indexed C0 through C15.
