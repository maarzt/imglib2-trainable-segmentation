
package net.imglib2.trainable_segmentation.utils.views;

import net.imglib2.Localizable;
import net.imglib2.RandomAccess;
import net.imglib2.Sampler;
import net.imglib2.util.Cast;
import net.imglib2.view.composite.Composite;
import net.imglib2.loops.ClassCopyProvider;

public class CompositeRandomAccess<T> implements RandomAccess<Composite<T>>, Composite<T> {

	private static final ClassCopyProvider<RandomAccess> provide =
		new ClassCopyProvider<>(CompositeRandomAccess.class, RandomAccess.class);

	public static <T> RandomAccess<Composite<T>> create(RandomAccess<T> ra) {
		return Cast.unchecked(provide.newInstanceForKey(ra.getClass(), ra));
	}

	private final RandomAccess<T> ra;

	private final int n;

	public CompositeRandomAccess(RandomAccess<T> ra) {
		this.ra = ra;
		this.n = ra.numDimensions() - 1;
	}

	@Override
	public RandomAccess<Composite<T>> copyRandomAccess() {
		return new CompositeRandomAccess<>(ra.copyRandomAccess());
	}

	@Override
	public Composite<T> get() {
		return this;
	}

	@Override
	public Sampler<Composite<T>> copy() {
		return copyRandomAccess();
	}

	@Override
	public T get(long i) {
		ra.setPosition(i, n);
		return ra.get();
	}

	@Override
	public long getLongPosition(int d) {
		return ra.getLongPosition(d);
	}

	@Override
	public void fwd(int d) {
		ra.fwd(d);
	}

	@Override
	public void bck(int d) {
		ra.bck(d);
	}

	@Override
	public void move(int distance, int d) {
		ra.move(distance, d);
	}

	@Override
	public void move(long distance, int d) {
		ra.move(distance, d);
	}

	@Override
	public void move(Localizable distance) {
		for (int d = 0; d < n; d++)
			move(distance.getLongPosition(d), d);
	}

	@Override
	public void move(int[] distance) {
		for (int d = 0; d < n; d++)
			move(distance[d], d);

	}

	@Override
	public void move(long[] distance) {
		for (int d = 0; d < n; d++)
			move(distance[d], d);
	}

	@Override
	public void setPosition(Localizable position) {
		for (int d = 0; d < n; d++)
			setPosition(position.getLongPosition(d), d);
	}

	@Override
	public void setPosition(int[] position) {
		for (int d = 0; d < n; d++)
			setPosition(position[d], d);
	}

	@Override
	public void setPosition(long[] position) {
		for (int d = 0; d < n; d++)
			setPosition(position[d], d);
	}

	@Override
	public void setPosition(int position, int d) {
		ra.setPosition(position, d);
	}

	@Override
	public void setPosition(long position, int d) {
		ra.setPosition(position, d);
	}

	@Override
	public int numDimensions() {
		return n;
	}
}
