
package net.imglib2.trainable_segmentation.pixel_feature.filter.gradient;

import org.junit.Ignore;
import org.junit.Test;
import net.imglib2.algorithm.convolution.kernel.Kernel1D;

import java.util.function.DoubleUnaryOperator;
import java.util.stream.DoubleStream;

import static org.junit.Assert.assertEquals;

/**
 * @author Matthias Arzt
 */
@Deprecated
public class DerivedNormalDistributionTest {

	@Test
	public void testDerivedNormalDistribution() {
		DoubleUnaryOperator f = DerivedNormalDistribution.derivedNormalDistribution(5);
		assertEquals(0.9718373972, f.applyAsDouble(2.0), 0.00001);
		assertEquals(-1.4518243471, f.applyAsDouble(1.0), 0.00001);
		assertEquals(0.0, f.applyAsDouble(0.0), 0.00001);
	}

	@Ignore("DerivedGaussKernel returns bad values for small sigma")
	@Test
	public void testDerivedGaussKernel() {
		Kernel1D kernel = DerivedNormalDistribution.derivedGaussKernel(0.3, 0);
		double sum = DoubleStream.of(kernel.fullKernel()).sum();
		assertEquals(1.0, sum, 0.0000001);
	}

}
