package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.lexer.TokenType;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenKind.BINARY;
import static com.deep.framework.ast.lexer.TokenType.*;

@Data
public class AssignExpression extends Expression {
    private Expression target;

    private Expression value;

    private TokenType operator;
    private static AssignExpression expression;
    static List listx = List.of(ASSIGN, ADD_ASSIGN, SUB_ASSIGN, MUL_ASSIGN, DIV_ASSIGN, AND_ASSIGN, OR_ASSIGN, XOR_ASSIGN, MOD_ASSIGN, LSHIFT_ASSIGN, RSHIFT_ASSIGN, URSHIFT_ASSIGN);

    public AssignExpression(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((List list, Object m, Object n, Object o) -> {
            if (Objects.nonNull(m) && Objects.nonNull(n) && Objects.nonNull(o)) {
                if (listx.contains(n) && list.size() == 2) {
                    expression = new AssignExpression(node);
                    expression.getChildrens().addAll(List.of(m, o));

                    Expression a = (Expression) m;
                    Expression c = (Expression) o;

                    a.setPrarent(expression);
                    c.setPrarent(expression);

                    expression.setTarget(a);
                    expression.setOperator((TokenType) n);
                    expression.setValue(c);

                    node.replace(m, expression);
                    node.getChildrens().remove(n);
                    node.getChildrens().remove(o);
                    list.remove(n);
                    list.remove(o);
                }
            }
        });
    }

}
