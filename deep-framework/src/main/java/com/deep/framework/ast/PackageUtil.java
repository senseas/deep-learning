package com.deep.framework.ast;

import java.io.File;
import java.lang.reflect.Modifier;
import java.net.JarURLConnection;
import java.net.URL;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

import static com.deep.framework.ast.lexer.TokenType.IMPORT;
import static com.deep.framework.ast.lexer.TokenType.PACKAGE;

public class PackageUtil {

    //获取类包下的所有子类
    public static <T> Set<Class<? extends T>> getSubTypesOf(Class<T> type) {
        Set<Class> classes = getAllClass(type.getPackage().getName());
        Set<Class<? extends T>> collect = new HashSet<>();
        for (Class aClass : classes) {
            if (!type.equals(aClass) && type.isAssignableFrom(aClass) && !aClass.isInterface() && !Modifier.isAbstract(aClass.getModifiers())) {
                collect.add((Class<? extends T>) aClass);
            }
        }
        return collect;
    }

    //获取某个包下的所有类
    public static Set<Class> getClass(String packageName) {
        Set<Class> classSet = new HashSet<>();
        try {
            String sourcePath = packageName.replace(".", File.separator);
            Enumeration<URL> urls = Thread.currentThread().getContextClassLoader().getResources(sourcePath);
            while (urls.hasMoreElements()) {
                URL url = urls.nextElement();
                if (Objects.nonNull(url)) {
                    String protocol = url.getProtocol();
                    if ("file".equals(protocol)) {
                        String packagePath = url.getPath().replaceAll("%20", " ");
                        addClass(classSet, packagePath, packageName);
                    } else if ("jar".equals(protocol)) {
                        JarURLConnection jarURLConnection = (JarURLConnection) url.openConnection();
                        if (jarURLConnection != null) {
                            JarFile jarFile = jarURLConnection.getJarFile();
                            if (jarFile != null) {
                                Enumeration<JarEntry> jarEntries = jarFile.entries();
                                while (jarEntries.hasMoreElements()) {
                                    String jarEntryName = jarEntries.nextElement().getName();
                                    if (jarEntryName.contains(sourcePath) && jarEntryName.endsWith(".class")) {
                                        String className = jarEntryName.substring(0, jarEntryName.lastIndexOf(".")).replaceAll(File.separator, ".");
                                        doAddClass(classSet, className);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return classSet;
    }

    //获取某个包下的所有类
    public static Set<Class> getAllClass(String packageName) {
        Set<Class> classSet = new HashSet<>();
        try {
            String sourcePath = packageName.replace(".", File.separator);
            Enumeration<URL> urls = Thread.currentThread().getContextClassLoader().getResources(sourcePath);
            while (urls.hasMoreElements()) {
                URL url = urls.nextElement();
                if (Objects.nonNull(url)) {
                    String protocol = url.getProtocol();
                    if ("file".equals(protocol)) {
                        String packagePath = url.getPath().replaceAll("%20", " ");
                        addAllClass(classSet, packagePath, packageName);
                    } else if ("jar".equals(protocol)) {
                        JarURLConnection jarURLConnection = (JarURLConnection) url.openConnection();
                        if (jarURLConnection != null) {
                            JarFile jarFile = jarURLConnection.getJarFile();
                            if (jarFile != null) {
                                Enumeration<JarEntry> jarEntries = jarFile.entries();
                                while (jarEntries.hasMoreElements()) {
                                    String jarEntryName = jarEntries.nextElement().getName();
                                    if (jarEntryName.contains(sourcePath) && jarEntryName.endsWith(".class")) {
                                        String className = jarEntryName.substring(0, jarEntryName.lastIndexOf(".")).replaceAll(File.separator, ".");
                                        doAddClass(classSet, className);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return classSet;
    }

    private static void addClass(Set<Class> classSet, String packagePath, String packageName) {
        File[] files = new File(packagePath).listFiles(file -> (file.isFile() && file.getName().endsWith(".class")) || file.isDirectory());
        for (File file : files) {
            String fileName = file.getName();
            if (file.isFile()) {
                String className = fileName.substring(0, fileName.lastIndexOf("."));
                if (isNotBlank(packageName)) {
                    className = packageName + "." + className;
                }
                doAddClass(classSet, className);
            }
        }
    }

    private static void addAllClass(Set<Class> classSet, String packagePath, String packageName) {
        File[] files = new File(packagePath).listFiles(file -> (file.isFile() && file.getName().endsWith(".class")) || file.isDirectory());
        for (File file : files) {
            String fileName = file.getName();
            if (file.isFile()) {
                String className = fileName.substring(0, fileName.lastIndexOf("."));
                if (isNotBlank(packageName)) {
                    className = packageName + "." + className;
                }
                doAddClass(classSet, className);
            } else {
                String subPackagePath = fileName;
                if (isNotBlank(packagePath)) {
                    subPackagePath = packagePath + File.separator + subPackagePath;
                }
                String subPackageName = fileName;
                if (isNotBlank(packageName)) {
                    subPackageName = packageName + "." + subPackageName;
                }
                addAllClass(classSet, subPackagePath, subPackageName);
            }
        }
    }

    public static Class loadClass(String className, boolean isInitialized) {
        Class clas;
        try {
            clas = Class.forName(className, isInitialized, Thread.currentThread().getContextClassLoader());
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
        return clas;
    }

    public static void forName(Node node) {
        String pkg = null;
        try {
            pkg = getPackage(node);
            Class clas = Class.forName(pkg);
            CompilationUnit.map.put(node.toString(), clas);
        } catch (ClassNotFoundException e) {
            CompilationUnit.pkg.add(pkg);
        }
    }

    //加载类（默认将初始化类）
    public static Class loadClass(String className) {
        return loadClass(className, true);
    }

    private static void doAddClass(Set<Class> classSet, String className) {
        Class clas = loadClass(className, false);
        classSet.add(clas);
    }

    private static boolean isNotBlank(String packageName) {
        return Objects.nonNull(packageName) && !packageName.isEmpty();
    }

    public static String getPackage(Node node) {
        for (Node n : node.getChildrens()) {
            if (!Stream.of(PACKAGE, IMPORT).contains(n.getTokenType())) {
                String name = getPackage(n);
                if (node.toString().startsWith("*")) return name;
                return name.concat(".").concat(node.toString());
            }
        }
        return node.toString();
    }

}