
package net.imglib2.trainable_segmentation.pixel_feature.filter.stats;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.test.ImgLib2Assert;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.view.Views;
import org.junit.Test;
import net.imglib2.algorithm.convolution.Convolution;
import net.imglib2.algorithm.convolution.LineConvolution;

/**
 * Tests {@link SumConvolver}.
 */
public class SumConvolverTest {

	@Test
	public void test() {
		Img<DoubleType> in = ArrayImgs.doubles(new double[] { 1, 2, 3, 4, 5, 6 }, 6);
		Img<DoubleType> out = ArrayImgs.doubles(5);
		Img<DoubleType> expected = ArrayImgs.doubles(new double[] { 3, 5, 7, 9, 11 }, 5);
		SumConvolver convolver = new SumConvolver(2, in.randomAccess(), out.randomAccess(), 0, out
			.dimension(0));
		convolver.run();
		ImgLib2Assert.assertImageEquals(expected, out);
	}

	@Test
	public void test2() {
		Img<DoubleType> in = ArrayImgs.doubles(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 10);
		Img<DoubleType> out = ArrayImgs.doubles(4);
		Img<DoubleType> expected = ArrayImgs.doubles(new double[] { 28, 35, 42, 49 }, 4);
		SumConvolver convolver = new SumConvolver(7, in.randomAccess(), out.randomAccess(), 0, out
			.dimension(0));
		convolver.run();
		ImgLib2Assert.assertImageEquals(expected, out);
	}

	@Test
	public void testC() {
		Img<DoubleType> input = ArrayImgs.doubles(new double[] {
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 0
		}, 5, 5);
		Img<DoubleType> output = ArrayImgs.doubles(5, 5);
		Img<DoubleType> expected = ArrayImgs.doubles(new double[] {
			0, 0, 0, 0, 0,
			0, 1, 1, 1, 0,
			0, 1, 2, 2, 1,
			0, 1, 2, 2, 1,
			0, 0, 1, 1, 1
		}, 5, 5);
		LineConvolution<DoubleType> convoltionX = new LineConvolution<>(SumConvolver.factory(1, 1), 0);
		LineConvolution<DoubleType> convoltionY = new LineConvolution<>(SumConvolver.factory(1, 1), 1);
		Convolution<DoubleType> convolution = Convolution.concat(convoltionX, convoltionY);
		convolution.process(Views.extendBorder(input), output);
		ImgLib2Assert.assertImageEquals(expected, output);
	}
}
