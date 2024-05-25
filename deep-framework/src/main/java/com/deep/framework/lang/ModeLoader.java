package com.deep.framework.lang;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class ModeLoader {

    public static void save(Object obj, String name) {
        String src = DataLoader.MODEL_PATH.concat(name);
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(src))) {
            out.writeObject(obj);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static <E> E load(String name) {
        String src = DataLoader.MODEL_PATH.concat(name);
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(src))) {
            Object o = in.readObject();
            return (E) o;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

}
