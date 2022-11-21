package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.FOR;
import static com.deep.framework.ast.lexer.TokenType.WHILE;

@Data
public class WhileStatement extends Statement {
    private Expression condition;
    public ParametersExpression parameters;
    private Statement body;
    private static WhileStatement statement;

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, m, n) -> {
            if (m.getChildrens().contains(WHILE)) {
                statement.setPrarent(node);
                statement.setChildrens(m.getChildrens());
                statement.remove(WHILE);
                if (Objects.nonNull(n) && n instanceof BlockStatement) {
                    statement.getChildrens().add(n);
                    statement.setBody((Statement) n);
                    node.replaceAndRemove(m, statement, n);
                    list.remove(n);
                } else {
                    node.replace(m, statement);
                }
            }
        });
    }
}
