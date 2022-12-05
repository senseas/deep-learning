package com.deep.framework.ast;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class CompilationUnit extends Node {
    public static Set<Class> set = new HashSet<>();
    public static Set<String> pkg = new HashSet<>();
    public static Map<String, Class> map = new HashMap<>();
    public static Object[] arr = new Object[1];
}