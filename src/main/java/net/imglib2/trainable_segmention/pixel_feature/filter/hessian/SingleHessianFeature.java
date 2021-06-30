
package net.imglib2.trainable_segmention.pixel_feature.filter.hessian;

import net.imglib2.*;
import net.imglib2.trainable_segmentation.RevampUtils;
import net.imglib2.trainable_segmentation.pixel_feature.filter.AbstractFeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureInput;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.settings.GlobalSettings;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.composite.Composite;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import net.imglib2.loops.LoopBuilder;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Matthias Arzt
 */
@Deprecated
@Plugin(type = FeatureOp.class, label = "Hessian")
public class SingleHessianFeature extends AbstractFeatureOp {

	@Parameter
	private double sigma = 1.0;

	@Override
	public int count() {
		return 8;
	}

	@Override
	public void apply(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		calculateHessianOnChannel(input.original(), output, sigma);
	}

	List<String> LABELS = Arrays.asList("", "_Trace", "_Determinant", "_Eigenvalue_1",
		"_Eigenvalue_2",
		"_Orientation", "_Square_Eigenvalue_Difference", "_Normalized_Eigenvalue_Difference");

	@Override
	public List<String> attributeLabels() {
		return LABELS.stream().map(x -> "Hessian" + x + "_" + sigma).collect(Collectors.toList());
	}

	@Override
	public boolean checkGlobalSettings(GlobalSettings globals) {
		return globals.numDimensions() == 2;
	}

	private static final int HESSIAN = 0;
	private static final int TRACE = 1;
	private static final int DETERMINANT = 2;
	private static final int EIGENVALUE_1 = 3;
	private static final int EIGENVALUE_2 = 4;
	private static final int ORIENTATION = 5;
	private static final int SQUARE_EIGENVALUE_DIFFERENCE = 6;
	private static final int NORMALIZED_EIGENVALUE_DIFFERENCE = 7;

	private void calculateHessianOnChannel(RandomAccessible<FloatType> image,
		List<RandomAccessibleInterval<FloatType>> output, double sigma)
	{
		double[] sigmas = { 0.4 * sigma, 0.4 * sigma };

		Interval secondDerivativeInterval = output.get(0);
		Interval firstDerivativeInterval = Intervals.union(
			RevampUtils.deriveXRequiredInput(secondDerivativeInterval), RevampUtils.deriveYRequiredInput(
				secondDerivativeInterval));
		Interval blurredInterval = Intervals.union(
			RevampUtils.deriveXRequiredInput(firstDerivativeInterval), RevampUtils.deriveYRequiredInput(
				firstDerivativeInterval));

		RandomAccessibleInterval<FloatType> blurred = RevampUtils.gauss(image, blurredInterval,
			sigmas);
		RandomAccessibleInterval<FloatType> dx = RevampUtils.deriveX(blurred,
			firstDerivativeInterval);
		RandomAccessibleInterval<FloatType> dy = RevampUtils.deriveY(blurred,
			firstDerivativeInterval);
		RandomAccessibleInterval<FloatType> dxx = RevampUtils.deriveX(dx,
			secondDerivativeInterval);
		RandomAccessibleInterval<FloatType> dxy = RevampUtils.deriveY(dx,
			secondDerivativeInterval);
		RandomAccessibleInterval<FloatType> dyy = RevampUtils.deriveY(dy,
			secondDerivativeInterval);

		LoopBuilder.setImages(RevampUtils.vectorizeStack(output), dyy, dxy, dxx).forEachPixel(
			(o, s_xx, s_xy, s_yy) -> calculateHessianPerPixel(o, s_xx.getRealFloat(), s_xy.getRealFloat(),
				s_yy.getRealFloat()));
	}

	private static void calculateHessianPerPixel(Composite<FloatType> output,
		float s_xx, float s_xy, float s_yy)
	{
		final double t = Math.pow(1, 0.75);

		// Hessian module: sqrt (a^2 + b*c + d^2)
		output.get(HESSIAN).set((float) Math.sqrt(s_xx * s_xx + s_xy * s_xy + s_yy * s_yy));
		// Trace: a + d
		final float trace = s_xx + s_yy;
		output.get(TRACE).set(trace);
		// Determinant: a*d - c*b
		final float determinant = s_xx * s_yy - s_xy * s_xy;
		output.get(DETERMINANT).set(determinant);

		// Ratio
		// ipRatio.set((float)(trace*trace) / determinant);
		// First eigenvalue: (a + d) / 2 + sqrt( ( 4*b^2 + (a - d)^2) / 2 )
		output.get(EIGENVALUE_1).set((float) (trace / 2.0 + Math.sqrt((4 * s_xy * s_xy + (s_xx - s_yy) *
			(s_xx - s_yy)) / 2.0)));
		// Second eigenvalue: (a + d) / 2 - sqrt( ( 4*b^2 + (a - d)^2) / 2 )
		output.get(EIGENVALUE_2).set((float) (trace / 2.0 - Math.sqrt((4 * s_xy * s_xy + (s_xx - s_yy) *
			(s_xx - s_yy)) / 2.0)));
		// Orientation
		if (s_xy < 0.0) // -0.5 * acos( (a-d) / sqrt( 4*b^2 + (a - d)^2)) )
		{
			float orientation = (float) (-0.5 * Math.acos((s_xx - s_yy) / Math.sqrt(4.0 * s_xy * s_xy +
				(s_xx - s_yy) * (s_xx - s_yy))));
			if (Float.isNaN(orientation))
				orientation = 0;
			output.get(ORIENTATION).set(orientation);
		}
		else // 0.5 * acos( (a-d) / sqrt( 4*b^2 + (a - d)^2)) )
		{
			float orientation = (float) (0.5 * Math.acos((s_xx - s_yy) / Math.sqrt(4.0 * s_xy * s_xy +
				(s_xx - s_yy) * (s_xx - s_yy))));
			if (Float.isNaN(orientation))
				orientation = 0;
			output.get(ORIENTATION).set(orientation);
		}
		// Gamma-normalized square eigenvalue difference
		output.get(SQUARE_EIGENVALUE_DIFFERENCE).set((float) (Math.pow(t, 4) * trace * trace * ((s_xx -
			s_yy) * (s_xx - s_yy) + 4 * s_xy * s_xy)));
		// Square of Gamma-normalized eigenvalue difference
		output.get(NORMALIZED_EIGENVALUE_DIFFERENCE).set((float) (Math.pow(t, 2) * ((s_xx - s_yy) *
			(s_xx - s_yy) + 4 * s_xy * s_xy)));
	}
}
