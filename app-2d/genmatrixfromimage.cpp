#include <iostream>
#include <Magick++.h>

int main(int argc, char** argv) {
        Magick::InitializeMagick(*argv);
	Magick::Image image(argv[1]);

	int w, h;

	w = image.columns();
	h = image.rows();

	if(w > h)
		w = h;
	else
		h = w;

	std::cout<<w;

	Magick::Pixels view(image);

	Magick::PixelPacket *pixels = view.get(0,0,w,h);

	for(int i = 0; i < w; i++) {
		for(int j = 0; j < h; j++) {
			Magick::ColorRGB color(pixels[w * i + j]);

			int r = (int)(color.red()*255);
			int g = (int)(color.green()*255);
			int b = (int)(color.blue()*255);

			int gray = (int)(0.21*r + 0.72*g + 0.07*b) - 127;
			//int gray = (int)(0.3*r + 0.59*g + 0.11*b) - 127;

			std::cout<<" "<<gray;
		}
	}

	return 0;
}
