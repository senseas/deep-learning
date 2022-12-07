package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.node.ArrayAccessNode;

import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.NEW;

public class ArrayAccessExpression extends Expression implements ArrayAccessNode {
    private Expression expression;
    private Expression index;

    public ArrayAccessExpression(Node prarent, Expression expression, Expression index) {
        super(prarent);
        this.expression = expression;
        this.index = index;

        this.expression.setPrarent(this);
        this.index.setPrarent(this);

        getChildrens().addAll(expression, index);
    }

    public static void parser(Node node) {
        if (node.getChildrens().stream().anyMatch(a -> a.equals(NEW))) return;
        Stream.of(node.getChildrens()).reduce2((list, a, b) -> {
            if (Objects.nonNull(b) && b instanceof ArrayExpression c && !c.getChildrens().isEmpty()) {
                Expression index = (Expression) c.getChildrens().get(0);
                ArrayAccessExpression arrayAccess = new ArrayAccessExpression(node, (Expression) a, index);
                list.replace(a, arrayAccess);
                list.remove(b);
            }
        });
    }

    public Expression getExpression() {
        return expression;
    }

    public Expression getIndex() {
        return index;
    }

    @Override
    public String toString() {
        return expression.toString().concat("[").concat(index.toString()).concat("]");
    }
}