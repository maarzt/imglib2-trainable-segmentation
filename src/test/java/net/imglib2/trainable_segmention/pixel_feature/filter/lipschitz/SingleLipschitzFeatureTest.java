
package net.imglib2.trainable_segmention.pixel_feature.filter.lipschitz;

import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.trainable_segmentation.Utils;
import net.imglib2.trainable_segmentation.pixel_feature.calculator.FeatureCalculator;
import net.imglib2.trainable_segmentation.pixel_feature.filter.SingleFeatures;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import org.junit.Test;
import net.imglib2.loops.LoopBuilder;

import java.util.function.DoubleBinaryOperator;

@Deprecated
public class SingleLipschitzFeatureTest {

	@Test
	public void test() {
		Img<FloatType> image = dirac();
		Img<FloatType> result = ArrayImgs.floats(5, 5);
		FeatureCalculator calculator = FeatureCalculator.default2d()
			.addFeature(SingleFeatures.lipschitz(0.1, 0))
			.build();
		calculator.apply(image, Views.addDimension(result, 0, 0));
		LoopBuilder.setImages(result).forEachPixel(x -> x.sub(new FloatType(255)));
		Img<FloatType> expected = createImage((x, y) -> (x == 2) && (y == 2) ? 0 : -1 + 0.1 * Math.sqrt(
			Math.pow(x - 2, 2) + Math.pow(y - 2, 2)));
		Utils.assertImagesEqual(35, expected, result);
	}

	@Test
	public void testOpening() {
		Img<FloatType> image = dirac();
		Img<FloatType> expected = createImage((x, y) -> 1 - Math.sqrt(Math.pow(0.1 * (x - 2), 2) + Math
			.pow(0.2 * (y - 2), 2)));
		ConeMorphology.performConeOperation(ConeMorphology.Operation.DILATION, image, new double[] {
			0.1, 0.2 });
		Utils.assertImagesEqual(35, expected, image);
	}

	@Test
	public void testClosing() {
		Img<FloatType> image = negate(dirac());
		Img<FloatType> expected = createImage((x, y) -> -1 + Math.sqrt(Math.pow(0.1 * (x - 2), 2) + Math
			.pow(0.2 * (y - 2), 2)));
		ConeMorphology.performConeOperation(ConeMorphology.Operation.EROSION, image, new double[] { 0.1,
			0.2 });
		Utils.assertImagesEqual(35, expected, image);
	}

	// -- Helper methods --

	private static Img<FloatType> dirac() {
		return createImage((x, y) -> (x == 2) && (y == 2) ? 1 : 0);
	}

	private static Img<FloatType> createImage(DoubleBinaryOperator function) {
		Img<FloatType> expected = ArrayImgs.floats(5, 5);
		Cursor<FloatType> cursor = expected.cursor();
		while (cursor.hasNext()) {
			cursor.fwd();
			double x = cursor.getDoublePosition(0);
			double y = cursor.getDoublePosition(1);
			cursor.get().setReal(function.applyAsDouble(x, y));
		}
		return expected;
	}

	private Img<FloatType> negate(Img<FloatType> dirac) {
		dirac.forEach(x -> x.mul(-1));
		return dirac;
	}

}
