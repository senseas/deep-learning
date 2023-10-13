package com.deep.framework.lang.util;

import com.huaban.analysis.jieba.JiebaSegmenter;

import java.util.List;
import java.util.stream.IntStream;

public class Sequence {

    private static String str = "在本文中，我们将试图把模型简化一点，并逐一介绍里面的核心概念，希望让普通读者也能轻易理解。";
    private static final JiebaSegmenter jieba_segmenter = new JiebaSegmenter();
    private static List<String> wordTable = getWordTable(str);

    public static List<double[]> getWordList(String str) {
        return jieba_segmenter
            .process(str, JiebaSegmenter.SegMode.INDEX)
            .stream().map(a -> oneHot(wordTable.indexOf(a.word)))
            .toList();
    }

    public static double[] getWordIndexList(String str) {
        return jieba_segmenter
            .process(str, JiebaSegmenter.SegMode.INDEX)
            .stream().mapToDouble(a -> wordTable.indexOf(a.word)).toArray();
    }

    public static List<String> getWordTable(String str) {
        return jieba_segmenter
            .process(str, JiebaSegmenter.SegMode.INDEX)
            .stream().map(a -> a.word)
            .distinct().sorted().toList();
    }

    public static double[] oneHot(int idx) {
        return IntStream.range(0, wordTable.size())
            .mapToDouble(a -> a == idx ? 1 : 0)
            .toArray();
    }

}
