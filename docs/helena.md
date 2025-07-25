# Abstract

Semiconductor pixel radiation detectors are widely used in the natural, physical, and medical sciences. For applications requiring tagging of individual particles with better than 10 µm spatial resolution and less than 1 µs temporal resolution, but with sparse hit occupancy and relaxed energy resolution, event-based counting detectors remain a reliable and common choice. By contrast, when particles need not be individually differentiated (in time or space), hit-rates are above 100 MHz/cm (without external trigger), and spatial resolution below 10 um are paramount, frame-based integrating-mode readouts are better-suited.

These frame-based architectures are especially advantageous for imaging-modes used in electron microscopy and X-ray imaging. In an experimental setting, fast frame rates allow for real time in-situ observations. Potential subjects include rapid chemical processes, molecular dynamics of proteins, crystal nucleation and growth, material phase transitions, thermal conductivity, charge transfer, and mechanical strain.

Our work pursues the possibility of a single-reticle array exceeding 1 Mpixel with a continuous frame-rate surpassing 100,000 fps, which would be unprecedented. For the conjunction of these two specifications to be met, this presentation will present initial investigations into a compact and power efficient bank of column-parallel data converters, which at 10-12 bit resolution churn out a combined data rate exceeding 1000 Gbps. To fit within the constraints of a chip-bottom, the converter fabric must consume less than 1 W/cm^2 and have an effective ADC sampling rate density above 5 ksps/μm^2. Successive approximation ADCs are identified as the optimal choice, and various topologies and techniques will be analyzed to meet our goals.

Extras:
ADCs are also used in the case of

Historically, the highest frame rates haves been achieved by buffering and duty cycle readout, either in-pixel in the analog domain, or in-matrix or in the chip periphery in the digital domain.

There are multiple components which together to determine the read out rate, which in-term deterine the frame. These including clock distribution, column sequencer, global shutter in-pixel transistor, clolumn drain line, column buffer, and ADC, and serializer, and serial driver, and wireline inteface. Of these, in most applications areas typically the most critic limiting factor is the

# Terminology

## Switching schemes (reducing power)

- Conventional (essentially [McCreary 1975](library/1975_James_McCreary_conventional_0%.pdf), with differential config)
- Split capacitor switching technique [2005 Brian Ginsburg](library/2005_Brian_Ginsburg-split_cap37%-charge_sharing24%-two_step10%.pdf) Showed three switching stragegies for reducing energy: 2-step, charge sharing, and split capacitor). The best was 'split capacitor', with a 37% energy reduction.
- Monotonic switching (aka 'set and down') [2010 C.C. Liu](library/2010_Chun-Cheng_Liu_monotonic_aka_set&down_switching_81%.pdf) 86% energy reduction. Only allows 'H->L' transistions on the plates. One disadvantage is continuous dropping of common mode.
- MCS: Merged capacitor switching [2010 Hariprasath](<library/2010_Venkatram_Hariprasath_merged_cap_switching_93%_(Vcm-based).pdf>)
- Vcm-based: [Yan Zhu 2010](library/2010_Yan_Zhu_Vcm-based_switching_87%.pdf), 87% reduction
- IMCS: Inverted merged capacitor swithing [2013 Albert Hsu](library/2013_Albert_Hsu_MIT_disseration_SAR_calibration.pdf)
- BSS: Bidirectional single-side switching [2014 Long Chen](library/2014_Long_Chen_bidirectional_singleside_switching.pdf) Is it similar to monotonic, except that the MSB starts at the opposite polarity to center the common-mode better.
- CAS: Charge-average switching [2013 C.C.Hsieh](library/2013_CY_Liou_CCHsieh_charge_average_switching_CAS.pdf)
- CRS: Correlated reverse switching [2015 Jen-Huan Tsai](library/2015_Jen-Huan_Tsai_correlated_reverse_switching_wADEC.pdf) Similar to BSS, but splits up the MSB cap?
- Merge-and-split [2015 Jin-Yi Lin](library/2015_Jin-Yi_Lin_merge&split_switching.pdf)

## Capacitor sizing

- capacitance splitting: switching half the effective value of each C_N on the P and N plate, to maintain common mode
-

## Redundancy

Redundancy can either be implemented in the CDAC itself, or in an arithmetic block which takes the comparator result to then determine a binary code which is closest in

I think these are all basically the same, in their redundancy benefits:

- non-binary, sub-radix 2. F. Kuttner 2005 does it with comp > arithmetic block -> nearest quantized binary value -> thermometer code
- "double conversion/multi conversion" is covered in CC Liu 2010, as "binary compensation redundancy". In
- "post-conversion" is covered in CC Liu 20

- [2014 Franz Kuttner](library/2004_Franz_Kuttner_subradix2_nonbinary.pdf): Is first to use redundancy via non-binary subradix 2
- [2013 Boris Murman](library/2013_Boris_Murmann_non_binary_survey.pdf):
- [2014 T. Waho](library/2014_T_Waho_nonbinary_redundancy_survey.pdf):
- [2015 C.C. Liu](library/2015_CC_Liu_redundancy.pdf):

## Digital error correction (DEC)

- Terms:

  - Addition-only DEC
  - adaptive equalization
  -

- [Wenbo Liu 2007 ISCAS Urbana](https://doi.org/10.1109/ICASIC.2007.4415624): An Equalization-Based Adaptive Digital Background Calibration Technique for Successive Approximation Analog-to-Digital Converters
- [Wenbo Liu 2009 ISSCC Urbana](https://doi.org/10.1109/ISSCC.2009.4977318): Implementation of above, with less detail

## Other terms

- Latch offet. [Murmann 2006](https://doi.org/10.1109/TCSII.2006.883204) analyzed regenerative latch offset from load cap mismatch
- NOTE: Split capacitor array (SCA) != split capacitor switching technique
- Top & bottom plate sampling
- Top & bottm plate switching
- Noise minimum shift (Caeleste, pg 26) An alternative to reference voltage regulation. When CDAC values are close to mid-scale, an equal number of capacitors are connected to Vhigh vs Vlow reference voltages, and the sensitivity to noise on the reference voltage is minimized. This technique add an additional capacitor array which of the input to minimized noise in each measurement.
- Split capacitors (statically biased caps, Caeleste pg58, p116). Similar to above, but divides caps into two, and sets them to different biases to improve noise
- Interleaving: Multiple ADCs operating in parrallel, sampling out of phase. Mainly necessary for >100Ms/s, at the expense of power and area. Not useful for our applications.
- Pipelining: Multiple sub-ADCs, operating sequentially. Similarly allows for greating speed, at higher power and area costs. Not necessary for our applications.

# Design techniques

- ✅ Unary, double conversion, and non-binary steps (typically x1.8 smaller)
- ✅ RDAC for powering lower voltage LSBs vs attenuating cap
- ❕ MoM != MIM, and POD is best @ 4fF in μm²
- ❌ MoM caps are only 0.6-0.8 fF in the same μm², with worse variation
- ❕ Most non-linearity comes from capacitor mismatch, but isn't a huge deal
- ❕ Gain loss comes from parasitic capacitance
- ✅ Double sample, and use differential cap array
- ✅ Auto-zero comparator, and use a basic dynamic/clocked latch design
- ❌ Async jeapordizes stability of neighbors
- ✅ Share sequencer across all ADCs
- ❌ Interleaving typically not worth it
- ✅ 7-8 Gbps JESD204B w/ 8b/10b encoding

# Capacitors types

You can create capacitor plates with poly, metal, and diffusion.

Typically the common combinations are:

- Poly-diffusion, aka MOS varactor or 'MOSCAP' with poly over n-wells (actually implanted not diffused). Typically the term Varactor is used in RF applications, where the structure will be done differenclty. ITRS 2011 puts this at 7-11 fF/μm²
- Poly-diffusion cap, where you use a field oxide instead of the standard FET gate
- 'Accumulation cap' which is a poly-diffusion (POD) cap, but with the S and D connection. Only using G to Bulk capacitance
- MIM caps, are created with metal layers, but place an intermetidate metal layer closer, and use a special insulator (silicon nitride, tantalum pentoxide Insulators (e.g., Si3N4 or Ta2O5)). ITRS 2011 puts this at 5-7 fF/μm²
- Metal-metal, aka metal-fringe-capacitor (MFC), availabe in TJ180. These can be multiple layers, or just a single layer. There are several variations, with intedigitated fingers with via between, or with vias only on the edges, and fingers rotated each and other layers called an 'RTMOM'. In TSMC65, this is covered [here](/eda/kits/TSMC/65LP/2024/V1.7A_1/1p9m6x1z1u/PDK_doc/TSMC_DOC_WM/PDK/crtmom_rf_device_route_guidance_for_RF_application.pdf). ITRS 2011 puts this at 5-7 fF/μm²?
- Poly-poly caps can be as simple as two large rectangles, on two seperate poly layers. Called 'POP' caps. Need to have a poly2 layer.

Tower 65nm seems to have nmoscap, moscaps, pmos caps (all these are varactors), mimcap, mimpcap 3T, RTMOM, and MIM. List is [here](/eda/kits/TSMC/65LP/2024/V1.7A_1/1p9m6x1z1u/PDK_doc/TSMC_DOC_WM/PDK/CRN65LP_v1d7a_pdkFSAChecklist.pdf).

Tower 180nm Accumulation caps, MIM caps, and MFCs are covered [here](/eda/kits/TOWER/ts18is_6M1L_2014/HOTCODE/models/ts18sl/v4.9.3/docs/DRS2_0018_B_manual.pdf).

A common way to increase the density of decoupling capacitors. Stack MOM on top of device capacitors (make sure they are connected in parallel and not series).

# Sep 23

cdac, resistive division, unitary vs
monotonic sar ADC, doulbe conversion, unitary capacitor
we have done binary and non-binary ADCs (aka 'redundancy', which allows you to ignore matching requirements, and go faster while saving space)
4-unary steps , followed by double conversion step (solution for dynamic error, by sampling twice).
At time zero, all comparators are brought to a high level., then one by one, each cap is lowered. This rejects power supply noise, and avoids the need for pregain even if you want different input ranges.

no ramp ADC, as SAR is better in energy and area per conversion per taime

power supply rejection is cucial, and

comparator runs on the clock, with a constant phase offset as it's redistributed to the

tower, xfab are preferred foundaries of caeleste

they use a simple, single stage comparator normally. For faster applications, an auto-zeroing comparator design could be good.

double conversion

Bonn has also tried segmented CDAC, capacitor back-side biasing.

Caeleste has tried regulation inside each ADC, but it's too big and consumpes too much power.

Speed 400 -> 2800 ADCs, goals is 10000 ADCs
rate 5Mhz, goal is 10Mhz
area 2500*1000Um -> 60x800um, goal 7.5um*200um
energy/conv 2.5nJ -> 155pj, 10pj goal

After the ADC:
subsequent step is to multiplex from ADC to JESD, then from JESD out to CML drives
in this case, the JESD actually uses more power than ADC banks.

The similarities between binary tracking detectors and optical light imagers is that neither are trying to perform high precision spectroscopy of the incident particles. In either case, you mostly know the energy of the particle before hand, and you job is to quantize their position and timing. With light, you'd like to achieve single particle sensitivity, or something close to it. With electrons, you has a good signal to noise, so you just need to worry about charge splitting etc so you require a bin size of maybe of just 25% of the 2-5ke- particle energy, and with high energy particles you typically just want to know if a single hit was above or below a single threshold.

The goal 7.5um\*200um goal is 1500um^2 of area, which is roughly half of a 50umx50um pixel. This means that compact SAR ADCs have similar areas in HEP and image sensors.

need: ledit, sedit, calibre, tspice,

tsl018
tpsc65

umc018
xc018

(lf15is used to be used)
lf11is

tsmc018
tsmc65

bug and optimization
12bit is bread a butter

cdac, sample and hold, comparator
we must stirve for lowe energy per conversion

area per converter, but not in terms of area

- excel
- include glue file
- library of generic .oa symbols
- exampline schematic

# Sep 23: Frame rate

```
from enum import Enum, auto

arch_access = "column_adj"  # other option is 'pixel_adj'

pixel_pitch = 10e-6 # should be on the order of micrometers
pixel_count = 2048*2048

ADC_speed = 10e6 # samples per second
ADC_bits = 8

# Calculate number of squares


W_metal = 50e-9 # should be 100's of nm, to 1's of um
L_metal =

R_sq = 0.001 # ohms per square


C = (A*L



# Using drive strength


# finally, calculate what percental of settling time will arrive on the bus, if you sample this quickly
```

# Sep 23: FOM

```
import math

def calc_sensor_dyn_range_dB(full_well_capacity: float, read_noise: float) -> float:

    return 10 * math.log10(full_well_capacity/read_noise)

def calc_sensor_dyn_range_linear(full_well_capacity: float, read_noise: float) -> float:

    return full_well_capacity/read_noise


print(dyn_range(100,1))


# read noise is the aggregate noise of
```

# Oct 14: ADC FOM

power / area (power density, < better)
conversions/ time / area (conversion density, > better)

energy / conversion

At low sampling speeds (

the Walden FOM appears is overly optimisitic at 'high resolution'. I think this surely includes a 12-16 SAR adc.

The Schreier FOM fontier at 10Msps (don't multiply by bits) is 185 dB. 160 is 'good' performance, but not SOTA. It begins to drop above 100Msps by -10dB / decade

The Walden FOM is simplye Fj/conversion-step. The SOTA frontier is around ~10 fJ per conversion-step, if below 100Msps Fsnyq. At this is F sampling nyquist

So like ~10 fj/conv-step is HARD and ~1 fJ/conv-step is best in the workd.

As ENOB increases, power typically is increased by 4x per incremental bit (Walden assumes only 2x)

# Oct 16: Helena

To do:

- Tasks:
  - behavioral DAC block is causing problem. Let's use Verilog-A or simulation math afterwards.
  - relative timing of clock vs nset vs comp_p and comp_n at cdac1_unit blocks. (charge suck out, from capacitance)
  - Fix 5 variations of ADC. Get all simulating on TSMC65 using Dropbox files.
  - Look for soundness of the timing above.
- Koen is python guy. koen liekens
- Python file for converting v2common.xlsx to .sp file? (Koen?)
- Sharing the ADC notes slides?
- Simulation testbench to properly quantify:
  - variations for application
  - monte-carlo simuations
  - arrays routing from sequencer
  - power supplies
- Design and synthesis of a streaming JESD204B encoder? Where to find verilog implementation?
- Helena acroynm? high speed low energy array HLEA
- Where are seq_dff_prog used?

ADC FOM:

- power / area (power density, less the better)
- conversions/ time / area (conversion density, greater the better)
- energy / conversion
- At low sampling speeds
- the Walden FOM appears is overly optimisitic at 'high resolution'. I think this surely includes a 12-16 SAR adc.
- The Schreier FOM fontier at 10Msps (don't multiply by bits) is 185 dB. 160 is 'good' performance, but not SOTA. It begins to drop above 100Msps by -10dB / decade
- The Walden FOM is simplye Fj/conversion-step. The SOTA frontier is around ~10 fJ per conversion-step, if below 100Msps Fsnyq. At this is F sampling nyquist
- So like ~10 fj/conv-step is HARD and ~1 fJ/conv-step is best in the workd.
- As ENOB increases, power typically is increased by 4x per incremental bit (Walden assumes only 2x)

Bart notes:

- Why 55 linear shrink of tsmc65? A certain customer wanted it.
- How are test benches stored? -> In the .oa file!
- All capacitor bias drivers should switch from high-to-low
- RDAC (optional) used to provide lower voltage bias to LSBs.
- you can have 1 or 2 buffers, and 2or3 sampling switches (the latter are if you reset the reference plate)
- How many unary + double conversion caps?
- Where to place unary + double conversion caps?
- Comparator auto-zeroing, and
- Sequencer now assumes reset, then sample. The reset+sample is "sort of" CDS?
- Modeling of capacitor mis-matches? -> Monte carlo mismatch
- Modeling of parasitics for speed and power consumption -> maybe not worth the effort. (Just assume 0.5x speed and 2x power?)
- Modeling of error creation (crosstalk, substrate noise, supply noise) + subsequent correction?
- NOTE: leaf symbols (which map to primitives) can't have user added params. Only heiarchicaly models which have an underlying schematic support these.
- Note that the in the alphamatized paramters section, the SPICE,OUTPUT section is
- Differential comparison algorithm
  - Pixel level is first reset by transferring into floating diffusion for short period
  - Then transfer gate is shut and pixel starts to integrated on diode while reset level is available on FD?
  - The reset level R is availabe on the column line which is applied to the input of one of the plates
  - Optionally, the reset level is written to both S and R plate, to allow for autozeroing (with buffer feedback in comp to allow auto-zeroing)
  - The pixel is then read, which produces a lower signal level S at the FD, which is buffered and fed to the other ADC plate

Amir notes:

- In our PDKs, MC models show 3-6% variation in min size cap mismatch.
- MoM != MIM.
  - MOM is on lower metal layers, and just uses fringe. Lowest density.
  - MIM is on higher layers, and uses a special dielectric material, not SiO2
- POD caps are linear FOEL devices, simialr to MOS caps but with a modified process step to add an extra layer.
  - They are the highest density, but must be supported by the foundary,
  - they are between poly and diffusion, but need sheidling on sides (Co, V1, v2) and above (from M1)
  - These additional caps add 'gain loss' though, as they reduce the voltage swing for a given Qin.
- Non linearity in INL comes from non-linear caps:
  - primarily the Cdb of the sample switch (drain-to-bulk depletion cap)
  - Also the gate-drain capacitor at input of comparator
  - but non-linearity isn't a huge deal, as pixel non-linearity dominates, and we can calibrate it anyways
  - This can be done across the whole analog chain, and given to customers as a software lookup table.
- Our unit caps are typically around 20 fF.
  - Determined by mismatch, power consumption, area limit, etc
- Freeing top metal layers is super useful for large arrays, as you can route power, control, clockers, sequencer data
- Tspice is quite slow. Eldo and AFS are not the same, but are similar in peformance. AFS just has RF built in.
- You can't realistically run transient noise simulation of an ADC. The best you can do is sweeping input transiently, and with monte-carlo mismatch.
- kTC noise
  - https://en.wikipedia.org/wiki/Johnson-Nyquist_noise#Thermal_noise_on_capacitors
  - Vn ~ sqrt(kT/C). Noise doesn't come from capacitor (cap has no noise), but does get converted to voltage by then.
  - If you have a input switch, you have one kTC, if you have an input buffer as well, it's noise figure adds an noise is 2 x kTC.
  - For small processes ( below 180nm), and low power design, you can assume you are kTC noise limited.
  - Small processses have better cap matching, so typically mismatch isn't dominant
  - If quantization noise is ~0.3LSB, then we just decrease VLSB until 0.2*LSB or 0.1*LSB of the kTC voltage noise.
  - ^^Past this point, noise will dominate so there is no point.
- SAR layout is typically long and tall. Input buffer+switch at top right. CDAC below-right. Sequencer+driver+decisionFFs on left of CDAC. Comparator small, at top left.
- In 180nm layout area is around 600um x 60um. We want to go much smaller though.

# Oct 14: adc.py

```
import math

# let's try to compute the FOM for M. Shin 2012. (ref 10)

def FoM(P, ConvTime, A_um2, DRDB):
    J_per_um2 = (P*ConvTime*A_um2)/(10**((DRDB-1.76)/10))
    fJ_per_um2 = J_per_um2*1e15
    return fJ_per_um2

# it has 2Mpixel, and 200Mpixels/second

# DCD adc convtime time is 1/((512*512*78000)/(16*16*8))=1e-7
# area of 200e-6*200e-6 = 4e-8 = 40000 um^2
# dynamic range is Qnoise limited, so 8*6.02+1.76 ~ 50 dB

DCD = FoM(3.6e-3,1e-7,40000,50)
print(DCD)
```

# Oct 15: Other notes

If you want to change models, how does that manifest in code?

And for models modifications like impact ionization, how does you submit the different continuity equation coefficients by B

Slide 54

For the breakdown simulation, what is the flatiron between the edge and backside potential? Slide 56

EDET system diagrams
What did CIX use for integrated charge ADC design. Is average energy useful?
What were the different apps of frame based readout within HEP? Why belle2 used it?

My research is system simulation(modeling) of radiation detector and imaging ASICs in CMOS processes to enable performance verification and optimization.

Radiation detectors and imaging asics can be differentiated based on

Along the way I have built know dogs in the following areas:

TCAD sensor simulation and

Obelix

1.5 W per chip
2x3 cm area
0.25 W / cm^2
33umx33um pixel

MightyPix1
TSI180
2x2 cm area
Targetting 0.15 W/cm^2 (Currently at 0.18 W/cm^2) achieved as there is no clock distribution, and readout is async along column drain
50umx150um pixel

MuPix10

EDET:
2uA \* 1.2V in area of 200um x 200um

Christian defense

Power consumption
Front end noise and threshold
Spatial resolution
Hit detection efficiency
Timing

30um high resistive epitaxial
100um highly resistive czharalski

165mW / cm2 power
(Atlas’s is 600mW)

The 8b10b en consign is used on chip

The beam telescope has a 3.7um with 25ns track timestamps

You want both MIP and high energy (5GeV?)

100um device looked like it was less efficiency? But labeled higher?

Is 99.7% pixel response just the rising edge, or the full ToT?
Time resolution is 1.25ns… that’s good? Dominated by the

100ps contribution from front end?

> 5\* 10^14 neutron NIeL damage, equivalent and > 1 MGy

For reference ALTAS can reach 10^16 NIEL

Dominated by clock distribution
Can do without clock distribution. (Like Malta)
