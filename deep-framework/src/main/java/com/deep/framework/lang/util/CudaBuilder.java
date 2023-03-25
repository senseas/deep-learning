package com.deep.framework.lang.util;

/**
 * @description:
 * @author: cdl
 * @create: 2021-03-25 17:54
 **/
public class CudaBuilder {

    public static String create(String src, Object... args) {
        try {
            byte[] bytes = CudaBuilder.class.getClassLoader().getResourceAsStream(src).readAllBytes();
            return new String(bytes).formatted(args);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

}