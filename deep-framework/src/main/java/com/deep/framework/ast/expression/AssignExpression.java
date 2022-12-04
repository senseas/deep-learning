package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.NodeList;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.lexer.TokenType;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.*;

@Data
public class AssignExpression extends Expression {
    private Name target;
    private Expression value;
    private TokenType operator;
    private static AssignExpression expression;
    private static List<TokenType> ASSIGN_TYPE = Stream.of(ASSIGN, ADD_ASSIGN, SUB_ASSIGN, MUL_ASSIGN, DIV_ASSIGN, AND_ASSIGN, OR_ASSIGN, XOR_ASSIGN, MOD_ASSIGN, LSHIFT_ASSIGN, RSHIFT_ASSIGN, URSHIFT_ASSIGN);

    public AssignExpression(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        ConditionalExpression.parser(node);
        Stream.of(node.getChildrens()).reduce((list, m, n, o) -> {
            if (Objects.nonNull(m) && Objects.nonNull(n) && Objects.nonNull(o)) {
                if (ASSIGN_TYPE.contains(n.getTokenType())) {
                    expression = new AssignExpression(node);
                    expression.setTarget((Name) m);
                    expression.setOperator(n.getTokenType());
                    expression.getChildrens().add(m);

                    NodeList<Expression> split = node.split(n);
                    if (Objects.nonNull(split)) {
                        Expression c = split.last();
                        c.setPrarent(expression);
                        expression.setValue(c);
                        expression.getChildrens().add(c);
                        node.getChildrens().remove(c);
                        node.getChildrens().removeAll(c.getChildrens());
                    }

                    node.getChildrens().remove(n);
                    node.replace(m, expression);
                    list.clear();
                }
            }
        });
    }

}