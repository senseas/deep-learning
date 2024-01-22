package com.deep.framework.lang.util;

import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.function.Func0;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Stack;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Tensorx {
    /**
     * 平铺树结构
     * @param root        节点树结构
     * @param getChildren 加载树的子节点列表函数 接收一个节点 返回节点的子结构
     * @param <M>         树节点对象
     * @return 平铺结构
     */
    public static <M> List<M> flatten(M[] root, Function<M, M[]> getChildren, Function<M, Tenser<M>> getFunction) {
        List<M> list = new ArrayList<>();
        forEach(root, getChildren, getFunction, list::add);
        return list;
    }

    /**
     * 平铺树结构
     * @param root        节点树结构
     * @param getChildren 加载树的子节点列表函数 接收一个节点 返回节点的子结构
     * @param <M>         树节点对象
     * @return 平铺结构
     */
    public static <M> List<M> flatten(M[] root, Function<M, M[]> getChildren) {
        List<M> list = new ArrayList<>();
        forEach(root, getChildren, list::add);
        return list;
    }

    /**
     * 聚合树结构
     * @param list        节点列表结构
     * @param getId       节点唯一key读取 接收一个节点 返回节点的唯一key
     * @param getParentId 节点父节点key读取 接收一个节点 返回节点的父节点key
     * @param setChildren 节点子项写入函数 接收待写入节点与节点子项 负责将子节点写入
     * @param <M>         节点对象
     * @param <N>         节点唯一key对象
     * @return 树结构
     */
    public static <M, N> List<M> gather(List<M> list, Function<M, N> getId, Function<M, N> getParentId, BiConsumer<M, M[]> setChildren) {
        List<M> root = list.stream().filter(o -> getParentId.apply(o) == null).collect(Collectors.toList());
        Stack<M> stack = new Stack<>();
        root.forEach(stack::push);
        while (!stack.isEmpty()) {
            M o = stack.pop();
            N id = getId.apply(o);
            List<M> children = list.stream().filter(m -> id.equals(getParentId.apply(m))).toList();
            if (Objects.nonNull(children)) {
                setChildren.accept(o, (M[]) children.toArray());
                children.forEach(stack::push);
            }
        }
        return root;
    }

    /**
     * 遍历树结构
     * @param root        节点树结构
     * @param getChildren 加载树的子节点列表函数 接收一个节点 返回节点的子结构
     * @param consumer    遍历到的节点行为
     * @param <M>         树节点对象
     */
    public static <M> void forEach(M[] root, Function<M, M[]> getChildren, Function<M, Tenser<M>> getFunction, Consumer<M> consumer) {
        Stack<M> stack = new Stack<>();
        for (M a : root) stack.push(a);
        while (!stack.isEmpty()) {
            M o = stack.pop();
            consumer.accept(o);
            M[] input = getChildren.apply(o);
            if (Objects.nonNull(input)) {
                for (M a : input) stack.add(0, a);
            }

            Tenser<M> function = getFunction.apply(o);
            if (Objects.nonNull(function)) {
                function.forEach(a -> stack.add(0, a));
            }
        }
    }

    /**
     * 遍历树结构
     * @param root        节点树结构
     * @param getChildren 加载树的子节点列表函数 接收一个节点 返回节点的子结构
     * @param consumer    遍历到的节点行为
     * @param <M>         树节点对象
     */
    public static <M> void forEach(M[] root, Function<M, M[]> getChildren, Func0<M> consumer) {
        Stack<M> stack = new Stack<>();
        for (M a : root) stack.push(a);
        while (!stack.isEmpty()) {
            M o = stack.pop();
            if (!consumer.apply(o)) {
                M[] input = getChildren.apply(o);
                if (Objects.nonNull(input)) {
                    for (M a : input) stack.push(a);
                }
            }
        }
    }
}