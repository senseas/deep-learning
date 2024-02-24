package com.deep.framework.lang.util;

import com.alibaba.fastjson2.JSONArray;
import com.huaban.analysis.jieba.JiebaSegmenter;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Sequence {
    private static final int batch_size = 500;
    private static final JiebaSegmenter jieba_segmenter = new JiebaSegmenter();
    private static List<String> wordTable = getTokentList();

    public static int getWordDicSize() {
        return wordTable.size();
    }

    public static List<double[]> getTokenOneHotList(String str) {
        return jieba_segmenter.process(str, JiebaSegmenter.SegMode.INDEX).stream().map(a -> oneHot(wordTable.indexOf(a.word))).toList();
    }

    public static List<String> getTokentList(String str) {
        return jieba_segmenter.process(str, JiebaSegmenter.SegMode.INDEX).stream().map(a -> a.word).distinct().sorted().toList();
    }

    public static double[] getTokenIndex(String str) {
        return jieba_segmenter.process(str, JiebaSegmenter.SegMode.INDEX).stream().mapToDouble(a -> wordTable.indexOf(a.word)).toArray();
    }

    public static double[] getTokenIndex(List<String> list) {
        return list.stream().mapToDouble(a -> wordTable.indexOf(a)).toArray();
    }

    public static double[] getFlatTokenOneHotList(List<String> list) {
        return list.stream().map(a -> oneHot(wordTable.indexOf(a))).flatMapToDouble(Arrays::stream).toArray();
    }

    public static double[] getWordIndex(String str) {
        return str.chars().mapToDouble(a -> wordTable.indexOf(String.valueOf((char) a).trim())).toArray();
    }

    public static List<String> getTokenList(String str) {
        List<String> strings = str.chars().mapToObj(a -> String.valueOf((char) a).trim()).collect(Collectors.toList());
        strings.add(0, "<begin>");
        strings.add("<end>");
        return strings;
    }

    public static double[] oneHot(int idx) {
        return IntStream.range(0, wordTable.size()).mapToDouble(a -> a == idx ? 1 : 0).toArray();
    }

    public static double[] oneHot(String str) {
        int idx = wordTable.indexOf(str);
        return IntStream.range(0, wordTable.size()).mapToDouble(a -> a == idx ? 1 : 0).toArray();
    }

    public static List<String> getTokentList() {
        byte[] bytes = null;
        try {
            bytes = FileUtil.readResourceAsStream("word.json").readAllBytes();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return JSONArray.parse(new String(bytes)).toJavaList(String.class);
    }

    public static List<String> getWordTable() {
        FileInputStream fileInputStreamWord = null;
        FileInputStream fileInputStreamTyc = null;
        try {
            fileInputStreamTyc = new FileInputStream("D:\\github\\chinese-xinhua-master\\data\\fh.txt");
            String[] tycText = new String(fileInputStreamTyc.readAllBytes()).split("\\n");
            List<String> word1 = Stream.of(tycText).map(String::trim).distinct().collect(Collectors.toList());

            fileInputStreamWord = new FileInputStream("D:\\github\\chinese-xinhua-master\\data\\word.json");
            String wordText = new String(fileInputStreamWord.readAllBytes());
            JSONArray parse = JSONArray.parse(wordText);
            List<String> word2 = parse.stream().map(a -> ((String) ((Map) a).get("word")).trim()).sorted().toList();
            word1.addAll(word2);
            return word1;
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                fileInputStreamWord.close();
                fileInputStreamTyc.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return new ArrayList<>();
    }

    public static List<List<String>> getMedicalTokenList() {
        FileInputStream fileInputStream = null;
        try {
            fileInputStream = new FileInputStream("/Users/chengdong/GitHub/deep-learning/deep-framework/src/main/resources/train_list.txt");
            String[] tycText = new String(fileInputStream.readAllBytes()).split("\\n");
            List<List<String>> words = Stream.of(tycText).map(s -> {
                String[] split = s.split("\\t");
                String str = split[1].concat(split[1].endsWith("?") ? "" : "?").concat(split[2]);
                return getTokenList(str);
            }).collect(Collectors.toList());
            return words;
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                fileInputStream.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return new ArrayList<>();
    }
}