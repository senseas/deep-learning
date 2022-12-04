package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;

import static com.deep.framework.ast.lexer.TokenType.BREAK;
import static com.deep.framework.ast.lexer.TokenType.CONTINUE;

public class ContinueStatement extends Statement {

    public ContinueStatement(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof SynchronizedStatement) return;
        Stream.of(node.getChildrens()).reduce2((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce2((c, m, n) -> {
                if (m.equals(CONTINUE)) {
                    //create SynchronizedNode and set Prarent，Parameters
                    ContinueStatement statement = new ContinueStatement(node);
                    //replace this node with SynchronizedNode
                    c.replace(m, statement);
                }
            });
        });
    }
}