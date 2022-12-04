package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;

import static com.deep.framework.ast.lexer.TokenType.BREAK;
import static com.deep.framework.ast.lexer.TokenType.RETURN;

public class BreakStatement extends Statement {

    public BreakStatement(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof SynchronizedStatement) return;
        Stream.of(node.getChildrens()).reduce2((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce2((c, m, n) -> {
                if (m.equals(BREAK)) {
                    //create SynchronizedNode and set Prarent，Parameters
                    BreakStatement statement = new BreakStatement(node);
                    //replace this node with SynchronizedNode
                    c.replace(m, statement);
                }
            });
        });
    }
}