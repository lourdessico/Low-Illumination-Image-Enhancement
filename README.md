# Low-Illumination-Image-Enhancement
Here I present a project on Low Illumination Image Enhancement based on:
 
_Z. Shi, M.m. Zhu, B.Guoet al. “Nighttime lowillumination image enhancement with single image using bright/dark channel prior.” EURASIPJournal on Image andVideoProcessingFeb, 2018, DOI.
 10.1186/s13640-018-0251-4_

## Authors
This project was made by Lourdes Simón and Oriol Josa.

## Brief summary

We reproduce a method of nighttime low-illumination image enhancement based on the use of bright and dark channels. Our starting point is an RGB image (I) with which we obtain a first approximation of the transmission (t) based on the bright channel (Ibright). Then we use the dark channel (Idark) to correct errors in the initial transmission estimate. Next, we apply a guided filter to improve the transmission. Finally, we obtain the well-exposed image (J) from the original image, the transmission, and an atmospheric light estimation (A). In the results obtained, we observe a clear improvement in the image’s exposure.

You can find more information and a detailed explanation of the method in the document called `Final Project on Low Illumination Image Enhancement - Lourdes and Oriol`.

At the end of the code you will find alternative ways to do some procedures that were discarted as we found a way which provided us with better or very similar results.

## Example Result
Below there is an example of our method results. More images can be found in the `images` folder.

<p align="center">
  <img src="images/camping.jpg" width="400"/><br>
  <img src="images/camp_results.jpg" width="400"/>
</p>

## Image License

The following images were created by **Lourdes Simón** and are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/):

- `images/laboratory.jpg`
- `images/lab_results.jpg`
- `images/light.jpg`
- `images/light_results.jpg`
- `images/plant.jpg`
- `images/plant_results.jpg`

You are free to:
- **Share** — copy and redistribute the images in any medium or format
- **Adapt** — remix, transform, and build upon the images

**Under the following terms:**
- **Attribution** — You must give appropriate credit to *Lourdes Simón*.
- **NonCommercial** — You may not use the images for commercial purposes.