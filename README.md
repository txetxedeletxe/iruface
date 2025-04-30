# Iruface
Create drawings and animations using complex fourier series to trace smooth paths (e.g. outlines of a picture).

### Fourier Coefficients
The following table summarizes the process to obtain Fourier Coefficients for the outline of a photo.

<div align="center">
  
| Photo | Edge Detection | Find Tour (Solve TSP) | Compute Fourier Coefficients|
|--------|--------|--------|--------|
| ![photo](/repo/img/iru.jpg) | ![edges](/repo/img/iruedge.png) | ![tour](/repo/img/tour.png) | ![fourier coefficients](/repo/img/coeff.png) |
  
</div>

### Drawings
Using Fourier Coefficients a doodle of the original outline can be obtained. Using more coefficients results in higher fidelity of the doodle as the Fourier Series approaches the source parametric curve, thus very high order fourier expansions are not as interesting artistically (and also require more compute resources).

<div align="center">
  
| Order 100 | Order 1000 | 
|--------|--------|
| ![fourier100](/repo/img/iruface_fourier_100.png) | ![fourier1000](/repo/img/iruface_fourier_1000.png) |

</div>

### Animations
The curve that the Fourier Series traces can be animated displaying in each timestep the contribution of each term in the series as a phasor. This allows visualizing the beautiful complexity of the Fourier Series in action.

<div align="center">
  
| Animation Frame | Zoomed in Frame | 
|--------|--------|
| ![animation](/repo/img/animation.png) | ![fourier1000](/repo/img/animation_zoom.png) |

</div>
