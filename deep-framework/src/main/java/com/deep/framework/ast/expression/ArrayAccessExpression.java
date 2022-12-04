package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import lombok.Data;

import java.util.Objects;

@Data
public class ArrayAccessExpression extends Expression {
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
        Stream.of(node.getChildrens()).reduce2((list, a, b) -> {
            if (a instanceof Name && Objects.nonNull(b) && b instanceof ArrayExpression c && !c.getChildrens().isEmpty()) {
                Expression index = (Expression) c.getChildrens().get(0);
                ArrayAccessExpression arrayAccess = new ArrayAccessExpression(node, (Expression) a, index);
                list.replace(a, arrayAccess);
                list.remove(b);
            }
        });
    }

    @Override
    public String toString() {
        return expression.toString().concat("[").concat(index.toString()).concat("]");
    }
}