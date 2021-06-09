
package net.imglib2.trainable_segmentation.gpu.algorithms;

import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.test.ImgLib2Assert;
import net.imglib2.trainable_segmentation.RevampUtils;
import net.imglib2.trainable_segmentation.Utils;
import net.imglib2.trainable_segmentation.gpu.api.AbstractGpuTest;
import net.imglib2.trainable_segmentation.gpu.api.GpuApi;
import net.imglib2.trainable_segmentation.gpu.api.GpuImage;
import net.imglib2.trainable_segmentation.gpu.api.GpuPool;
import net.imglib2.trainable_segmentation.gpu.api.GpuViews;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.junit.Test;
import net.imglib2.algorithm.gauss3.Gauss3;

import static org.junit.Assume.assumeTrue;

/**
 * Tests {@link GpuGauss}.
 */
public class GpuGaussTest extends AbstractGpuTest {

	@Test
	public void test() {
		RandomAccessible<FloatType> dirac = Utils.dirac(3);
		RandomAccessibleInterval<FloatType> expected = RevampUtils.createImage(Intervals.createMinMax(
			-2, -2, -2, 2, 2, 2), new FloatType());
		Gauss3.gauss(2, dirac, expected);
		Interval targetInterval = new FinalInterval(expected);
		GpuNeighborhoodOperation operation = GpuGauss.gauss(gpu, 2, 2, 2);
		Interval inputInterval = operation.getRequiredInputInterval(targetInterval);
		try (
			GpuImage input = gpu.push(Views.interval(dirac, inputInterval));
			GpuImage output = gpu.create(Intervals.dimensionsAsLongArray(targetInterval),
				NativeTypeEnum.Float);)
		{
			operation.apply(GpuViews.wrap(input), GpuViews.wrap(output));
			RandomAccessibleInterval<FloatType> rai = gpu.pullRAI(output);
			ImgLib2Assert.assertImageEqualsRealType(Views.zeroMin(expected), rai, 1.e-7);
		}
	}
}
