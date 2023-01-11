


#include "fftwTest.h"
#include <stdio.h>
/* load original image */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/video/video.hpp>  
#include <opencv2/videoio/videoio.hpp>  
#include <opencv2/videoio/legacy/constants_c.h>  
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>  

#if defined(_DEBUG)
#pragma comment(lib, "/x64/vc15/lib/opencv_world451d.lib")
#else
#pragma comment(lib, "/x64/vc15/lib/opencv_world451.lib")
#endif

using namespace cv;

int fftwImageTest()
{
	Mat img_src = imread("D:/resource/images/lena_g.png", IMREAD_GRAYSCALE);
	if (img_src.empty())
	{
		std::cout << "cannot load file" << std::endl;
		return 0;
	}

	/* create new image for FFT & IFFT result */
	Mat img_fft = Mat::zeros(Size(img_src.cols, img_src.rows), CV_32FC1);
	Mat img_ifft = Mat::zeros(Size(img_src.cols, img_src.rows), CV_32FC1);

	/* get image properties */
	int width = img_src.cols;
	int height = img_src.rows;

	/* initialize arrays for fftw operations */
	fftw_complex *data_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * width * height);
	fftw_complex *fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * width * height);
	fftw_complex *ifft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * width * height);

	/* create plans */
	fftw_plan plan_f = fftw_plan_dft_2d(height, width, data_in, fft, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan plan_b = fftw_plan_dft_2d(height, width, fft, ifft, FFTW_BACKWARD, FFTW_ESTIMATE);

	img_src.convertTo(img_src, CV_32FC1);
	imwrite("original_image.jpg", img_src);
	int i, j, k;
	/* load img_src's data to fftw input */
	for (i = 0, k = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			data_in[k][0] = img_src.at<float>(i, j);
			data_in[k][1] = 0.0;
			k++;
		}
	}

	/* perform FFT */
	fftw_execute(plan_f);

	/* perform IFFT */
	fftw_execute(plan_b);

	/* normalize FFT result */
	double maxx = 0.0, minn = 10000000000.0;
	for (i = 0; i < width * height; ++i)
	{
		fft[i][0] = log(sqrt(fft[i][0] * fft[i][0] + fft[i][1] * fft[i][1]));
		maxx = fft[i][0] > maxx ? fft[i][0] : maxx;
		minn = fft[i][0] < minn ? fft[i][0] : minn;
	}

	for (i = 0; i < width * height; ++i)
	{
		fft[i][0] = 255.0 * (fft[i][0] - minn) / (maxx - minn);
	}

	/* copy FFT result to img_fft's data */
	int i0, j0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (i < height / 2)
				i0 = i + height / 2;
			else
				i0 = i - height / 2;
			if (j < width / 2)
				j0 = j + width / 2;   // method 2
			else
				j0 = j - width / 2;

			img_fft.at<float>(i, j) = (float)fft[/*k++*/i0 * width + j0][0];
		}
	}

	/* normalize IFFT result */
	for (i = 0; i < width * height; ++i)
	{
		ifft[i][0] /= width * height;
	}

	/* copy IFFT result to img_ifft's data */
	for (i = 0, k = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			img_ifft.at<float>(i, j) = (float)ifft[k++][0];
		}
	}

	/* display images */
	imwrite("FFT.jpg", img_fft);
	imwrite("IFFT.jpg", img_ifft);

	/* free memory */
	fftw_destroy_plan(plan_f);
	fftw_destroy_plan(plan_b);
	fftw_free(data_in);
	fftw_free(fft);
	fftw_free(ifft);
	return 0;
}