package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;

import static com.deep.framework.ast.lexer.TokenType.RETURN;

public class ReturnStatement extends Statement {
    private Expression expression;

    public ReturnStatement(Node prarent, Expression expression) {
        super(prarent);
        this.expression = expression;
        this.expression.setPrarent(this);
        getChildrens().addAll(expression);
    }

    public static void parser(Node node) {
        if (node instanceof SynchronizedStatement) return;
        Stream.of(node.getChildrens()).reduce2((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce2((c, m, n) -> {
                if (m.equals(RETURN)) {
                    c.remove(m);
                    //create SynchronizedNode and set Prarent，Parameters
                    ReturnStatement statement = new ReturnStatement(node, (Expression) a);
                    //replace this node with SynchronizedNode
                    list.replace(a, statement);
                }
            });
        });
    }
}