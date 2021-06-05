package com.deep.framework.lang.util;

import java.io.InputStream;

/**
 * @description:
 * @author: cdl
 * @create: 2021-03-25 17:54
 **/
public class FileUtil {

    public static InputStream readResourceAsStream(String src) {
       try{
            return FileUtil.class.getClassLoader().getResourceAsStream(src);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

}
