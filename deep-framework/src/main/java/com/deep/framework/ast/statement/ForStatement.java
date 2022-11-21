package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.FOR;

@Data
public class ForStatement extends Statement {
    private List<Expression> initialization;
    private Expression compare;
    private List<Expression> update;

    public ParametersExpression parameters;
    private Statement body;
    private static ForStatement statement;

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, m, n) -> {
            if (m.getChildrens().contains(FOR)) {
                statement = new ForStatement();
                statement.setPrarent(node);
                statement.setChildrens(m.getChildrens());
                statement.remove(FOR);
                if (Objects.nonNull(n) && n instanceof BlockStatement) {
                    statement.getChildrens().add(n);
                    statement.setBody((Statement) n);
                    ((Node) n).setPrarent(statement);
                    node.replaceAndRemove(m, statement, n);
                    list.remove(n);
                } else {
                    node.replace(m, statement);
                }
            }
        });
    }
}