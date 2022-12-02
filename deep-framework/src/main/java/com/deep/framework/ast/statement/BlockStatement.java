package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.NodeList;

public class BlockStatement extends Statement {
    public BlockStatement(Node prarent, NodeList<Node> childrens) {
        super(prarent);
        childrens.forEach(a -> a.setPrarent(this));
        getChildrens().addAll(childrens);
    }
}