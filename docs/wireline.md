- IBIS models (vs spice models)
- RLGC models
- touchstone files
- W models
- Telegrapher's equations (only for TEM modes)
- S-parameters vs Z-parameters
- Vector network analyzer vs time domain reflectometer
- Transmission lines modeled as lumped or distributed elements


distrubted element filters, power dividers, and circulators are made of stubs, coupled lines, cascaded lines, etc

The RLGC telegrapher's equations are valid for TEM transmission lines. Although microstrip lines are not purely TEM, the error is minimal and RLGC models are still used. This is referred to as quasi-TEM.

lossless vs lossy lines

W-element is RLGC element in SPICE (Wxxxx naming), uses per-unit-length parameters (PUL)
and also supports values at different frequencies (since conductor loss increaes with frequency
due to skin effect and dielectric loss is frequency dependent as well)

Tools like Simbeor can generate RGLC models with any of the three available field solvers (SFS, 3DML and 3DTF), with the SFS solver used most often

Signal and power integrity: SIPI analyses using industry-standard tools (e.g., Ansys SI-Wave, HFSS-3D-layout, and cadence tools




- SOTA drivers consume 1-10 pJ per bit, including the serializer and equalizer
- [BoW maxes out at 50mm](https://opencomputeproject.github.io/ODSA-BoW/bow_specification.html#sec-bow-modes)

Mapping of protocols to drivers and power consumption (via deepseek):

Interface,Modulation,Data Rate per Lane,TX Energy per Bit,Driver Type,Key Drivers of Power
PCIe 6.0,PAM4,64 GT/s (32 GBd),5–10 pJ/bit,CML,PAM4 DSP, FEC, adaptive equalization
PCIe 5.0,NRZ,32 GT/s,8–15 pJ/bit,CML,NRZ with heavy equalization (CTLE, DFE)
GDDR6X (PAM4),PAM4,21–24 Gbps,4–8 pJ/bit,CML (Proprietary),Optimized for GPU memory (lower protocol overhead)
400G Ethernet,PAM4,53.125 GBd (106.25Gbps),10–20 pJ/bit,CML,DSP-heavy PAM4, FEC, retimer logic
DDR5,NRZ (POD125),6.4 Gbps,2–5 pJ/bit,POD125,Parallel interface, simpler TX logic
InfiniBand NDR,PAM4,112 Gbps,8–12 pJ/bit,CML,Low-latency optimized PAM4 PHY
CXL 3.0 (PCIe 6),PAM4,64 GT/s,5–10 pJ/bit,CML,Inherits PCIe 6.0 PHY + CXL protocol overhead
