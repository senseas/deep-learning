package com.deep.framework.lang.util;

import static java.lang.Math.*;

final public class MathExt {

    public static double cot(double a) {
        return cos(a) / sin(a);
    }

    public static double sec(double a) {
        return 1 / cos(a);
    }

    public static double csc(double a) {
        return 1 / sin(a);
    }

    public static double arcsin(double a) {
        return asin(a);
    }

    public static double arccos(double a) {
        return acos(a);
    }

    public static double arctan(double a) {
        return atan(a);
    }

    public static double arccot(double a) {
        return arctan(1 / a);
    }

    public static double arcsec(double a) {
        return arccos(1 / a);
    }

    public static double arccsc(double a) {
        return arcsin(1 / a);
    }
}