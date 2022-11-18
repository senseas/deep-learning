package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.FOR;

public class ForEachStatement extends Statement {
    private List<Expression> initialization;
    private Expression compare;
    private List<Expression> update;
    private Statement body;
    private static ForStatement statement;

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((List list, Object m, Object n) -> {
            if (m instanceof Node a) {
                if (a.getChildrens().contains(FOR)) {
                    statement = new ForStatement();
                    statement.setPrarent(node);
                    statement.setChildrens(a.getChildrens());
                    statement.remove(FOR);
                    if (Objects.nonNull(n) && n instanceof BlockStatement) {
                        statement.getChildrens().add(n);
                        statement.setBody((Statement) n);
                        ((Node) n).setPrarent(statement);
                        node.replaceAndRemove(a, statement, n);
                        list.remove(n);
                    } else {
                        node.replace(a, statement);
                    }
                }
            }
        });
    }

}