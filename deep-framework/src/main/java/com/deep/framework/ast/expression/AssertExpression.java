package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.lexer.TokenType;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.ASSERT;

@Data
public class AssertExpression extends Expression {
    private Expression condition;
    private Expression detail;

    public AssertExpression(Node prarent, Expression condition, Expression detail) {
        super(prarent);
        this.condition = condition;
        this.detail = detail;

        this.condition.setPrarent(this);
        this.detail.setPrarent(this);

        getChildrens().addAll(condition, detail);
    }

    public static void parser(Node node) {
        ArrayAccessExpression.parser(node);
        Stream.of(node.getChildrens()).reduce((list, a, b, c, d) -> {
            if (a.equals(ASSERT) && Objects.nonNull(c) && c.equals(TokenType.COLON)) {
                AssertExpression expression = new AssertExpression(node, (Expression) b, (Expression) d);
                node.replace(a, expression);
                list.removeAll(List.of(b, c, d));
            }
        });
    }

    @Override
    public String toString() {
        return ASSERT.toString().concat(" ").concat(condition.toString()).concat(":").concat(detail.toString());
    }

}