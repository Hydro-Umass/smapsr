**Super-Resolution of SMAP Soil Moisture Satellite Observations using
Neural ODEs**

This repository contains code for a project that focuses on
super-resolving Soil Moisture Active Passive (SMAP) satellite
observations from 9-km to 3-km and 1-km resolutions. The algorithm
leverages the power of Neural Ordinary Differential Equations (Neural
ODEs) to achieve this goal.

**Background**
Soil moisture is a crucial parameter in understanding Earth's water
cycle, climate modeling, and agricultural applications. SMAP satellite
provides global soil moisture data at a resolution of 9-km, which can
be limiting for certain applications. To address this limitation, we
propose a novel approach that uses Neural ODEs to super-resolve the
9-km SMAP data to higher resolutions.

**Methodology**
The algorithm is trained on the SMAP/Sentinel-1 data product, which
provides a unique opportunity to leverage the strengths of both
sensors. The Neural ODE framework is used to model the complex
relationships between the SMAP and Sentinel-1 data, enabling the
generation of high-resolution soil moisture maps.

**Repository Structure**
This repository contains the following directories and files:

* `smapsr`: contains the implementation of the Neural ODE framework
* `env`: Nix flake files to reproduce the environment
