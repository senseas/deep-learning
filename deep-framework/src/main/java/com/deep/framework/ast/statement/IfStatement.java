package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.ELSE;
import static com.deep.framework.ast.lexer.TokenType.IF;

@Data
public class IfStatement extends Statement {
    private Expression condition;
    private List thenStatement;
    private Statement elseStatement;
    private Statement body;
    private static IfStatement statement;

    public static void parser(Node node) {
        if (node instanceof IfStatement) return;
        Stream.of(node.getChildrens()).reduce((list, m, n) -> {
            if (Objects.isNull(statement) && m.getChildrens().contains(IF)) {
                statement = new IfStatement();
                statement.setPrarent(node);
                statement.setChildrens(m.getChildrens());
                statement.remove(IF);
                if (Objects.nonNull(n) && n instanceof BlockStatement) {
                    statement.setBody((Statement) n);
                    node.remove(n);
                    statement.getChildrens().add(n);
                    ((Node) n).setPrarent(statement);
                    node.replaceAndRemove(m, statement, n);
                    list.remove(n);
                } else {
                    node.replace(m, statement);
                }
            } else if (m.getChildrens().contains(ELSE) && m.getChildrens().contains(IF)) {
                m.setPrarent(statement);
                statement.getChildrens().add(m);
                statement.remove(List.of(ELSE, IF));
                if (Objects.nonNull(n) && n instanceof BlockStatement) {
                    m.getChildrens().add(n);
                    ((Node) n).setPrarent(m);
                    node.getChildrens().removeAll(List.of(m, n));
                    list.remove(n);
                } else {
                    node.remove(m);
                }
            } else if (m.getChildrens().contains(ELSE)) {
                m.setPrarent(statement);
                statement.getChildrens().add(m);
                statement.remove(ELSE);
                if (Objects.nonNull(n) && n instanceof BlockStatement) {
                    m.getChildrens().add(n);
                    n.setPrarent(m);
                    node.getChildrens().removeAll(List.of(m, n));
                    list.remove(n);
                } else {
                    node.remove(m);
                }
                statement = null;
            }
        });
    }
}