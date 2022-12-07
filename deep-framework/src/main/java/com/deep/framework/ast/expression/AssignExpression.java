package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.node.AssignNode;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.*;

public class AssignExpression extends Expression implements AssignNode {
    private Expression variable;
    private Expression value;
    private TokenType operator;

    public AssignExpression(Node prarent, Expression variable, Expression value, TokenType operator) {
        super(prarent);
        this.variable = variable;
        this.value = value;
        this.operator = operator;

        this.variable.setPrarent(this);
        this.value.setPrarent(this);

        getChildrens().addAll(variable, value);
    }

    public static void parser(Node node) {
        ConditionalExpression.parser(node);
        Stream.of(node.getChildrens()).reduce((list, m, n, o) -> {
            if (Objects.nonNull(m) && Objects.nonNull(n) && Objects.nonNull(o)) {
                if (ASSIGN_TYPE.contains(n.getTokenType())) {
                    AssignExpression expression = new AssignExpression(node, (Expression) m, (Expression) o, n.getTokenType());
                    node.replaceAndRemove(m, expression, n);
                    node.getChildrens().remove(o);
                    list.removeAll(List.of(n, o));
                }
            }
        });
    }

    @Override
    public Expression getVariable() {
        return variable;
    }

    @Override
    public Expression getValue() {
        return value;
    }

    @Override
    public TokenType getOperator() {
        return operator;
    }

    @Override
    public String toString() {
        return variable.toString().concat(" ").concat(operator.toString()).concat(" ").concat(value.toString());
    }
}