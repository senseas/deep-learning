package com.deep.framework.ast;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.stream.Stream;

public class FileUtil {

    public static Stream<String> readFile(String strFile) {
        try (InputStream is = new FileInputStream(strFile)) {
            int iAvail = is.available();
            byte[] bytes = new byte[iAvail];
            is.read(bytes);
            return new String(bytes).chars().mapToObj(a -> String.valueOf((char) a));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

}
