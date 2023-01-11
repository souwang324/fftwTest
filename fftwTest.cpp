
#include "fftwTest.h"

int fftwTest() {

	fftw_complex x[5];
	fftw_complex y[5];

	for (int i = 0; i < 5; i++) {
		x[i][0] = i;
		x[i][1] = 0;
	}

	fftw_plan plan = fftw_plan_dft_1d(5, x, y, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_execute(plan);

	for (int i = 0; i < 5; i++) {
		cout << y[i][0] << "  " << y[i][1] << endl;
	}

	fftw_destroy_plan(plan);

	cout << "\nPress Enter to exit..." << endl;
	cin.get();

	return 0;
}