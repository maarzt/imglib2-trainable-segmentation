package net.imglib2.algorithm.features.gson;

import com.google.gson.Gson;
import com.google.gson.JsonElement;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;

/**
 * @author Matthias Arzt
 */
final public class GsonUtils {

	private GsonUtils() {
		// prevent from instantiation.
	}

	private Gson gson() {
		return new Gson();
	}

	void write(JsonElement json, String filename) {
		try(Writer writer = new FileWriter(filename)) {
			gson().toJson(json, writer);
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	JsonElement read(String filename) {
		try(Reader reader = new FileReader(filename)) {
			return gson().fromJson(reader, JsonElement.class);
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	void write(JsonElement json, OutputStream out) {
		try(Writer writer = new OutputStreamWriter(out)) {
			gson().toJson(json, writer);
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	JsonElement read(InputStream in) {
		try(Reader reader = new InputStreamReader(in)) {
			return gson().fromJson(reader, JsonElement.class);
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	String toString(JsonElement json) {
		try(Writer writer = new StringWriter()) {
			gson().toJson(json, writer);
			return writer.toString();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	JsonElement fromString(String text) {
		try(Reader reader = new StringReader(text)) {
			return gson().fromJson(reader, JsonElement.class);
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
}
