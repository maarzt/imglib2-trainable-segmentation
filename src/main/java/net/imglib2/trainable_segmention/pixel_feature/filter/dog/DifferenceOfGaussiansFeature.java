
package net.imglib2.trainable_segmention.pixel_feature.filter.dog;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.trainable_segmentation.pixel_feature.filter.AbstractFeatureOp;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureInput;
import net.imglib2.trainable_segmentation.pixel_feature.filter.FeatureOp;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Pair;
import net.imglib2.util.ValuePair;
import org.scijava.plugin.Plugin;
import net.imglib2.loops.LoopBuilder;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Matthias Arzt
 */
@Deprecated
@Plugin(type = FeatureOp.class, label = "Difference of Gaussians (Group)")
public class DifferenceOfGaussiansFeature extends AbstractFeatureOp {

	private List<Double> sigmas;
	private List<Pair<Double, Double>> sigmaPairs;

	private List<Pair<Double, Double>> sigmaPairs() {
		List<Pair<Double, Double>> sigmaPairs = new ArrayList<>();
		for (double sigma1 : sigmas)
			for (double sigma2 : sigmas)
				if (sigma2 < sigma1)
					sigmaPairs.add(new ValuePair<>(sigma1, sigma2));
		return sigmaPairs;
	}

	@Override
	public void initialize() {
		sigmas = globalSettings().sigmas();
		sigmaPairs = sigmaPairs();
	}

	@Override
	public int count() {
		return sigmaPairs.size();
	}

	@Override
	public List<String> attributeLabels() {
		return sigmaPairs.stream().map(pair -> "Difference_of_gaussians_" + pair.getA() + "_" + pair
			.getB())
			.collect(Collectors.toList());
	}

	@Override
	public void apply(FeatureInput input, List<RandomAccessibleInterval<FloatType>> output) {
		for (int i = 0; i < output.size(); i++) {
			Pair<Double, Double> sigma1and2 = sigmaPairs.get(i);
			RandomAccessibleInterval<FloatType> target = output.get(i);
			subtract(input.gauss(sigma1and2.getB() * 0.4), input.gauss(sigma1and2.getA() * 0.4), target);
		}
	}

	private static void subtract(RandomAccessibleInterval<? extends RealType<?>> minuend,
		RandomAccessibleInterval<? extends RealType<?>> subtrahend,
		RandomAccessibleInterval<FloatType> target)
	{
		LoopBuilder.setImages(minuend, subtrahend, target).multiThreaded().forEachPixel(
			(m, s, t) -> t.setReal(m.getRealFloat() - s.getRealFloat()));
	}
}
