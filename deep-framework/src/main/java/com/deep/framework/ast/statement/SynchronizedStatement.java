package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import static com.deep.framework.ast.lexer.TokenType.SYNCHRONIZED;

public class SynchronizedStatement extends Statement {
    private BlockStatement body;
    private Expression expression;

    public SynchronizedStatement(Node prarent, Expression expression, BlockStatement body) {
        super(prarent);
        this.expression = expression;
        this.body = body;

        this.expression.setPrarent(this);
        this.body.setPrarent(this);

        getChildrens().addAll(expression, body);
    }

    public static void parser(Node node) {
        if (node instanceof SynchronizedStatement) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(SYNCHRONIZED) && n instanceof ParametersExpression) {
                    //create SynchronizedNode and set Prarent，Parameters
                    SynchronizedStatement statement = new SynchronizedStatement(node, (Expression) n, (BlockStatement) b);
                    //replace this node with SynchronizedNode
                    node.replace(a, statement);
                    node.getChildrens().remove(b);
                    list.remove(b);
                }
            });
        });
    }

}