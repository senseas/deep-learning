package com.deep.framework.lang;

import java.io.*;

public class ModeLoader {

    public static void save(Object obj, String src) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(src))) {
            out.writeObject(obj);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static <E> E load(String src) {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(new File(src)))) {
            Object o = in.readObject();
            return (E) o;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

}
